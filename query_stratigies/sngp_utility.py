import copy
import torch
import torch.nn.functional as F
from model.sngp import mean_field_logits
from datetime import datetime

@torch.no_grad()
def get_gp_features(model, dataloader, device):
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
    cov = model.compute_predictive_covariance(phis.to(device))
    return phis, targets, cov

@torch.no_grad()
def get_prediction(model, device, phi, cov):
    model.eval()
    model.to(device)
    logits = model.beta(phi.to(device))
    scaled_logits = mean_field_logits(logits, cov, model.mean_field_factor)
    return scaled_logits

def get_probmatrix(model, device, phi):
    model.eval()
    model.to(device)
    logits_sampled = torch.einsum('nd,ekd->enk', phi.to(device), model.sampled_betas.to(device))
    return logits_sampled

@torch.no_grad()
def get_reweighted_model(model, phis, targets, lmb=1):
    model.eval()
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