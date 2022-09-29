import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger, SmoothedValue
from metrics import ood, generalization, calibration
from time import time


class SNGP(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 in_features: int,
                 num_classes: int,
                 num_inducing: int = 1024,
                 kernel_scale: float = 1,
                 normalize_input: bool = False,
                 scale_random_features: bool = False,
                 mean_field_factor: float = math.pi/8,
                 cov_momentum: float = -1,
                 ridge_penalty: float = 1,
                 ):
        super().__init__()
        self.model = model

        # in_features -> num_inducing -> num_classes
        self.in_features = in_features
        self.num_inducing = num_inducing
        self.num_classes = num_classes

        # Scale input
        self.kernel_scale = kernel_scale

        # Norm input
        self.normalize_input = normalize_input
        if self.normalize_input:
            self.kernel_scale = 1
            self.layer_norm = nn.LayerNorm(in_features)

        # Random features
        self.scale_random_features = scale_random_features

        # Inference
        self.mean_field_factor = mean_field_factor

        # Covariance computation
        self.cov_momentum = cov_momentum
        self.ridge_penalty = ridge_penalty

        self.random_features = RandomFourierFeatures(
            in_features=in_features,
            num_inducing=num_inducing,
            kernel_scale=self.kernel_scale,
            scale_features=self.scale_random_features
        )

        # Define output layer according to Eq 8., For imagenet init with normal std=0.01
        self.beta = nn.Linear(num_inducing, num_classes, bias=False)
        nn.init.xavier_normal_(self.beta.weight)

        # precision matrix
        self.init_precision_matrix = torch.eye(num_inducing)*self.ridge_penalty
        self.register_buffer("precision_matrix", copy.deepcopy(self.init_precision_matrix))
        self.cov_mat = None

        self.sampled_betas = None

    @property
    def covariance_matrix(self):
        device = self.precision_matrix.data.device
        if self.cov_mat is None:
            u = torch.linalg.cholesky(self.precision_matrix.data)
            self.cov_mat = torch.cholesky_inverse(u)
        return self.cov_mat.to(device)

    def reset_precision_matrix(self):
        device = self.precision_matrix.device
        self.precision_matrix.data = copy.deepcopy(self.init_precision_matrix)
        self.precision_matrix.to(device)
        self.cov_mat = None

    @torch.no_grad()
    def update_precision_matrix(self, phi, logits):
        probas = logits.softmax(-1)
        probas_max = probas.max(1)[0]
        precision_matrix_minibatch = torch.matmul(
            probas_max * (1-probas_max) * phi.T, phi
        )
        if self.cov_momentum > 0:
            batch_size = len(phi)
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (self.cov_momentum * self.precision_matrix.data +
                                    (1-self.cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix.data + precision_matrix_minibatch
        self.precision_matrix.data = precision_matrix_new
        self.cov_mat = None

    def compute_predictive_covariance(self, phi):
        covariance_matrix_feature = self.covariance_matrix
        out = torch.matmul(covariance_matrix_feature, phi.T) * self.ridge_penalty
        covariance_matrix_gp = torch.matmul(phi, out)
        return covariance_matrix_gp

    def forward(self, x, return_cov=False):
        _, features = self.model(x, return_features=True)

        if self.normalize_input:
            features = self.layer_norm(features)

        # Get gp features according to Eq. 7
        phi = self.random_features(features)

        # Eq. 8
        logits = self.beta(phi)

        if self.training:
            self.update_precision_matrix(phi, logits)
        if return_cov:
            cov = self.compute_predictive_covariance(phi)
            return logits, cov
        return logits

    @torch.no_grad()
    def forward_mean_field(self, x):
        if self.training:
            raise ValueError("Call eval mode before!")
        logits, cov = self.forward(x, return_cov=True)
        scaled_logits = mean_field_logits(logits, cov, self.mean_field_factor)
        return scaled_logits

    def sample_betas(self, n_draws):
        dist = torch.distributions.MultivariateNormal(
            loc=self.beta.weight,
            covariance_matrix=self.covariance_matrix
        )
        self.sampled_betas = dist.sample(sample_shape=(n_draws,))

    @torch.no_grad()
    def forward_sample(self, x):
        _, features = self.model(x, return_features=True)
        if self.normalize_input:
            features = self.layer_norm(features)
        phi = self.random_features(features)
        logits_sampled = torch.einsum('nd,ekd->enk', phi, self.sampled_betas)
        return logits_sampled

    def forward_dirichlet(self, x, use_variance_correction=False):
        if self.training:
            raise ValueError("Call eval mode before!")
        # Get logit mean and covariance predictions.
        logits, cov = self(x, return_cov=True)
        var = torch.diag(cov)
        var = torch.clamp(var, min=1.e-5)

        # Zero mean correction.
        logits -= ((var * logits.sum(-1)) / (var * self.num_classes))[:, None]
        var *= (self.num_classes - 1) / self.num_classes

        # Optional variance correction.
        if use_variance_correction:
            c = var / math.sqrt(self.num_classes/2)
            logits /= c.sqrt()[:, None]
            var /= c

        # Compute alphas.
        sum_exp = torch.exp(-logits).sum(dim=1).unsqueeze(-1)
        alphas = (1 - 2/self.num_classes + logits.exp()/self.num_classes**2 * sum_exp) / var[:, None]
        return alphas


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, num_inducing=1024, kernel_scale=1, scale_features=True):
        super().__init__()
        self.kernel_scale = kernel_scale
        self.input_scale = 1/math.sqrt(self.kernel_scale)

        self.scale_features = scale_features
        self.random_feature_scale = math.sqrt(2./float(num_inducing))

        self.random_feature_linear = nn.Linear(in_features, num_inducing)
        self.random_feature_linear.weight.requires_grad = False
        self.random_feature_linear.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self, std_init=1):
        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/resnet50_sngp.py#L55
        nn.init.normal_(self.random_feature_linear.weight, std=std_init)
        nn.init.uniform_(self.random_feature_linear.bias, 0, 2*math.pi)

    def forward(self, x):
        # Supports lengthscale for cutom random feature layer by directly rescaling the input.
        x = x * self.input_scale
        x = torch.cos(self.random_feature_linear(x))

        # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/wide_resnet_sngp.py#L207
        if self.scale_features:
            # Scale random feature by 2. / sqrt(num_inducing).  When using GP
            # layer as the output layer of a nerual network, it is recommended
            # to turn this scaling off to prevent it from changing the learning
            # rate to the hidden layers.
            x = self.random_feature_scale * x
        return x


def mean_field_logits(logits, cov, lmb=math.pi / 8):
    """Scale logits using the mean field approximation proposed by https://arxiv.org/abs/2006.07584"""
    if lmb is None or lmb < 0:
        return logits
    variances = torch.diag(cov).view(-1, 1) if cov is not None else 1
    logits_adjusted = logits / torch.sqrt(1 + lmb*variances)
    return logits_adjusted


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.reset_precision_matrix()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        acc1, = generalization.accuracy(outputs, targets, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats


@torch.no_grad()
def evaluate(model, dataloader_id, ood_loaders, criterion, device):
    test_stats = {}
    model.eval()
    model.to(device)

    # Forward categorical in distribution
    logits_id, targets_id, alphas_id, vars_id = [], [], [], []
    t = time()
    for inputs, targets in dataloader_id:
        inputs = inputs.to(device)
        logits_scaled = model.forward_mean_field(inputs).cpu()
        logits_id.append(logits_scaled)
        targets_id.append(targets)
    test_stats.update({
        'prediction_time': time() - t
    })
    logits_id = torch.cat(logits_id, dim=0)
    targets_id = torch.cat(targets_id, dim=0)

    # Forward dirichlet in distribution
    for inputs, targets in dataloader_id:
        inputs = inputs.to(device)
        alphas = model.forward_dirichlet(inputs)
        _, cov = model(inputs, return_cov=True)
        alphas_id.append(alphas.cpu())
        vars_id.append(torch.diag(cov).cpu())
    alphas_id = torch.cat(alphas_id, dim=0)
    vars_id = torch.cat(vars_id, dim=0)

    # Update test stats
    criterion_tce = calibration.TopLabelCalibrationError()
    criterion_mce = calibration.MarginalCalibrationError()
    criterion_bs = calibration.BrierScore()
    criterion_nll = nn.CrossEntropyLoss()

    # Generalization
    loss = criterion(logits_id, targets_id)
    acc1, = generalization.accuracy(logits_id, targets_id, (1,))
    test_stats.update({
        'loss': loss.item(),
        'acc1': acc1.item(),
    })

    # Calibration
    probas_id = logits_id.softmax(-1)
    nll = criterion_nll(logits_id, targets_id)
    brier_score = criterion_bs.forward(probas_id, targets_id)
    tce = criterion_tce(probas_id, targets_id)
    mce = criterion_mce(probas_id, targets_id)
    def tolist(x): return x.tolist() if isinstance(x, torch.Tensor) else x
    tcp = {k: tolist(v) for k, v in criterion_tce.results.items()}
    mcp = [{k: tolist(v) for k, v in curve.items()} for curve in criterion_mce.results]
    test_stats.update({
        'brier_score': brier_score.item(),
        'nll': nll.item(),
        'tce': tce.item(),
        'mce': mce.item(),
        'tcp': tcp,
        'mcp': mcp,
    })

    # Out-of-Distribution
    dist_id = torch.distributions.Dirichlet(alphas_id)
    dir_entropy = dist_id.entropy().mean()
    dir_variance = dist_id.variance.sum(-1).mean()
    dir_alpha_sum = dist_id.concentration.sum(-1).mean()
    test_stats.update({
        'dir_entropy': dir_entropy.item(),
        'dir_max_variance': dir_variance.item(),
        'dir_alpha_sum': dir_alpha_sum.item(),
    })

    for ds_name, ood_loader in ood_loaders.items():
        print(f'Evaluating OOD metrics for {ds_name}')
        # Forward prop out of distribution
        logits_ood, alphas_ood, vars_ood = [], [], []
        for inputs, _ in ood_loader:
            logits_ood.append(model.forward_mean_field(inputs.to(device)))
            alphas_ood.append(model.forward_dirichlet(inputs.to(device)))
            vars_ood.append(torch.diag(model(inputs.to(device), return_cov=True)[1]))
        logits_ood = torch.cat(logits_ood, dim=0).cpu()
        alphas_ood = torch.cat(alphas_ood, dim=0).cpu()
        vars_ood = torch.cat(vars_ood, dim=0).cpu()

        # Dirichlet
        dist_ood = torch.distributions.Dirichlet(alphas_ood)
        entropy_id, entropy_ood = dist_id.entropy(), dist_ood.entropy()
        test_stats.update({
            f'auroc_{ds_name}_dir_entropy': ood.ood_auroc(entropy_id, entropy_ood),
            f'aupr_{ds_name}_dir_entropy': ood.ood_aupr(entropy_id, entropy_ood),
        })
        alpha_sum_id, alpha_sum_ood = dist_id.concentration.sum(-1), dist_ood.concentration.sum(-1)
        test_stats.update({
            f'auroc_{ds_name}_dir_alpha_sum': ood.ood_auroc(-alpha_sum_id, -alpha_sum_ood),
            f'aupr_{ds_name}_dir_alpha_sum': ood.ood_aupr(-alpha_sum_id, -alpha_sum_ood),
        })
        variance_id, variance_ood = dist_id.variance.sum(-1), dist_ood.variance.sum(-1)
        test_stats.update({
            f'auroc_{ds_name}_dir_variance': ood.ood_auroc(variance_id, variance_ood),
            f'aupr_{ds_name}_dir_variance': ood.ood_aupr(variance_id, variance_ood),
        })

        # Categorical
        probas_ood = logits_ood.softmax(-1)
        entropy_id, entropy_ood = ood.entropy_fn(probas_id), ood.entropy_fn(probas_ood)
        test_stats.update({
            f'auroc_{ds_name}_cat_entropy': ood.ood_auroc(entropy_id, entropy_ood),
            f'aupr_{ds_name}_cat_entropy': ood.ood_aupr(entropy_id, entropy_ood),
        })
        test_stats.update({
            f'auroc_{ds_name}_cat_var': ood.ood_auroc(vars_id, vars_ood),
            f'aupr_{ds_name}_cat_var': ood.ood_aupr(vars_id, vars_ood),
        })

    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats


@torch.no_grad()
def reweight(model, dataloader, device, lmb=1):
    model.eval()
    model.to(device)

    # Get all features and targets
    all_phis, all_targets = [], []
    for inputs, targets in dataloader:
        _, features = model.model(inputs.to(device), return_features=True)
        if model.normalize_input:
            features = model.layer_norm(features)
        phi = model.random_features(features)
        all_phis.append(phi.cpu())
        all_targets.append(targets)
    phis = torch.cat(all_phis)
    targets = torch.cat(all_targets)

    # Reweight
    model.cpu()
    mean = model.beta.weight.data.clone()
    cov = model.covariance_matrix.data.clone()
    targets_onehot = F.one_hot(targets, num_classes=model.num_classes)

    for phi, target_onehot in zip(phis, targets_onehot):
        for _ in range(lmb):
            tmp_1 = cov @ phi
            tmp_2 = torch.outer(tmp_1, tmp_1)

            # Compute new prediction.
            var = F.linear(phi, tmp_1)
            logits = F.linear(phi, mean)
            probas = logits.softmax(-1)
            probas_max = probas.max()

            # Update covariance matrix.
            num = probas_max * (1-probas_max)
            denom = 1 + num * var
            factor = num / denom
            cov_update = factor * tmp_2
            cov -= cov_update

            # Update mean.
            tmp_3 = F.linear(cov, phi)
            tmp_4 = (target_onehot - probas)
            mean += torch.outer(tmp_4, tmp_3)

            # Undo cov update.
            cov += cov_update

            # Compute new prediction.
            var = F.linear(phi, tmp_1)
            logits = F.linear(phi, mean)
            probas = logits.softmax(-1)
            probas_max = probas.max()

            # Update covariance matrix.
            num = probas_max * (1 - probas_max)
            denom = 1 + num * var
            factor = num / denom
            cov_update = factor * tmp_2
            cov -= cov_update

    model_reweighted = copy.deepcopy(model)
    model_reweighted.beta.weight.data = mean
    model_reweighted.covariance_matrix.data = cov
    return model_reweighted


@torch.no_grad()
def evaluate_sampling(model, dataloader, ood_loaders, criterion, device, n_draws=1000, weights=None, sample_betas=True):
    test_stats = {}
    model.eval()
    model = model.to(device)

    logits_id, targets_id = [], []
    t = time()
    if sample_betas:
        model.sample_betas(n_draws=n_draws)
    for X_batch, y_batch in dataloader:
        logits = model.forward_sample(X_batch.to(device)).cpu()
        logits_id.append(logits)
        targets_id.append(y_batch)
    test_stats.update({
        'prediction_time': time() - t
    })
    logits_id = torch.cat(logits_id, dim=1)
    targets_id = torch.cat(targets_id)

    probas_id_raw = logits_id.softmax(-1)
    if weights is not None:
        probas_id = torch.einsum('e,enk->nk', weights, probas_id_raw)
        variance_id = torch.einsum('e,enk->nk', weights, (probas_id_raw - probas_id[None]) ** 2)
    else:
        probas_id = probas_id_raw.mean(0)
        variance_id = ((probas_id_raw - probas_id[None]) ** 2).mean(0)
    variance_id = variance_id.sum(-1)

    # Compute loss and accuracy
    eps = torch.finfo(probas_id.dtype).eps
    mean_logits_id = probas_id.clamp(eps, 1-eps).log()
    loss = criterion(mean_logits_id, targets_id)

    acc1, = generalization.accuracy(probas_id, targets_id)
    test_stats.update({
        'loss': loss.item(),
        'acc1': acc1.item(),
    })

    # Compute calibration metrics
    criterion_tce = calibration.TopLabelCalibrationError()
    criterion_mce = calibration.MarginalCalibrationError()
    criterion_bs = calibration.BrierScore()
    criterion_nll = calibration.NegativeLogLikelihood()

    nll = criterion_nll(probas_id, targets_id)
    brier_score = criterion_bs(probas_id, targets_id)
    tce = criterion_tce(probas_id, targets_id)
    mce = criterion_mce(probas_id, targets_id)
    def tolist(x): return x.tolist() if isinstance(x, torch.Tensor) else x
    tcp = {k: tolist(v) for k, v in criterion_tce.results.items()}
    mcp = [{k: tolist(v) for k, v in curve.items()} for curve in criterion_mce.results]

    test_stats.update({
        'brier_score': brier_score.item(),
        'nll': nll.item(),
        'tce': tce.item(),
        'mce': mce.item(),
        'tcp': tcp,
        'mcp': mcp,
    })

    for ds_name, ood_loader in ood_loaders.items():
        print(f'Evaluating OOD metrics for {ds_name}')

        # Forward prop out of distribution
        logits_ood = torch.cat([model.forward_sample(inputs.to(device)).cpu() for inputs, _ in ood_loader], dim=1)

        probas_ood_raw = logits_ood.softmax(-1)
        if weights is not None:
            probas_ood = torch.einsum('e,enk->nk', weights, probas_ood_raw)
            variance_ood = torch.einsum('e,enk->nk', weights, (probas_ood_raw - probas_ood[None]) ** 2)
        else:
            probas_ood = probas_ood_raw.mean(0)
            variance_ood = ((probas_ood_raw - probas_ood[None]) ** 2).mean(0)
        variance_ood = variance_ood.sum(-1)

        # Update test stats
        entropy_id = ood.entropy_fn(probas_id)
        entropy_ood = ood.entropy_fn(probas_ood)
        test_stats.update({
            f'auroc_{ds_name}_cat_entropy': ood.ood_auroc(entropy_id, entropy_ood),
            f'aupr_{ds_name}_cat_entropy': ood.ood_aupr(entropy_id, entropy_ood)
        })

        test_stats.update({
            f'auroc_{ds_name}_dir_variance': ood.ood_auroc(variance_id, variance_ood),
            f'aupr_{ds_name}_dir_variance': ood.ood_aupr(variance_id, variance_ood)
        })

    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats

@torch.no_grad()
def reweight_sampling(model, dataloader, device, n_draws=1000, return_params=False):
    model.eval()
    model.to(device)

    # Get all features and targets
    all_phis, all_targets = [], []
    for inputs, targets in dataloader:
        _, features = model.model(inputs.to(device), return_features=True)
        if model.normalize_input:
            features = model.layer_norm(features)
        phi = model.random_features(features)
        all_phis.append(phi)
        all_targets.append(targets)
    phis = torch.cat(all_phis)
    targets = torch.cat(all_targets)

    # Sample ensemble members.
    model.sample_betas(n_draws=n_draws)

    # Compute logits per sample and ensemble member.
    logits = torch.einsum('ekd,nd->enk', model.sampled_betas, phis).cpu()

    # Compute weights of ensemble members.
    _, n_samples, n_classes = logits.shape
    log_probas_sampled = logits.log_softmax(-1)
    log_prior = torch.log(torch.ones(n_draws) / n_draws)  # uniform prior
    log_likelihood = log_probas_sampled[:, range(n_samples), targets].sum(dim=1)
    log_weights = log_prior + log_likelihood
    weights = torch.exp(log_weights - log_weights.max())  # normalize log probs numerically stable
    weights /= weights.sum()

    if not return_params:
        return weights
    else:
        mean = torch.einsum('e,edk->dk', weights, model.sampled_betas)
        cov = [torch.cov(model.sampled_betas[:, i].T, correction=0, aweights=weights) for i in range(2)]
        cov = torch.stack(cov, dim=0).mean(0)
        return weights, mean, cov