import torch
import torch.nn as nn

from utils import MetricLogger, SmoothedValue
from metrics import generalization, ood, calibration
from time import time


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        pass

    def forward_sample(self, x):
        logits = []
        for m in self.models:
            logits.append(m(x))
        return torch.stack(logits)

    def __iter__(self):
        for m in self.models:
            yield m

    def __len__(self):
        return len(self.models)


class EnsembleOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def state_dict(self):
        return [optim.state_dict() for optim in self.optimizers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for optim, state_dict in zip(self.optimizers, state_dict_list):
            optim.load_state_dict(state_dict)

    def __iter__(self):
        for optim in self.optimizers:
            yield optim


class EnsembleLRScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def state_dict(self):
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def load_state_dict(self, state_dict_list: list) -> None:
        for scheduler, state_dict in zip(self.schedulers, state_dict_list):
            scheduler.load_state_dict(state_dict)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()


def train_one_epoch(models, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    for i_model, (model, optim) in enumerate(zip(models, optimizer)):
        model.train()
        model.to(device)

        metric_logger = MetricLogger(delimiter=" ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        header = f"Epoch [{epoch}] Model [{i_model}] " if epoch is not None else f"Model [{i_model}] "

        # Train the epoch
        for inputs, targets in metric_logger.log_every(dataloader, print_freq, header):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_size = inputs.shape[0]
            acc1, = generalization.accuracy(outputs, targets, topk=(1,))
            metric_logger.update(loss=loss.item(), lr=optim.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, ood_loaders, criterion, device, weights=None):
    test_stats = {}

    model = model.eval()
    model = model.to(device)

    logits_id, targets_id = [], []
    t = time()
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
def reweight(model, dataloader, device, lmb=1):
    model.eval()
    model.to(device)

    # Get all features and targets
    logits, targets = [], []
    for X_batch, y_batch in dataloader:
        logits.append(model.forward_sample(X_batch.to(device)).cpu())
        targets.append(y_batch)
    logits = torch.cat(logits, dim=1)
    targets = torch.cat(targets)

    draws, n_samples, _ = logits.shape
    log_probas_sampled = logits.log_softmax(-1)
    log_prior = torch.log(torch.ones(draws) / draws)  # uniform prior
    log_likelihood = log_probas_sampled[:, range(n_samples), targets].sum(dim=1)
    log_weights = log_prior + lmb*log_likelihood
    weights = torch.exp(log_weights - log_weights.max())  # normalize log probs numerically stable
    weights /= weights.sum()
    return weights
