import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MetricLogger, SmoothedValue
from metrics import generalization, ood, calibration
from .mc_dropout import MCDropoutModule, ConsistentMCDropout2d
from time import time


class DropoutBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, p_drop, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_drop = ConsistentMCDropout2d(p_drop)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_drop(out)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DropoutBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, p_drop, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv1_drop = ConsistentMCDropout2d(p_drop)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2_drop = ConsistentMCDropout2d(p_drop)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_drop(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.conv2_drop(out)
        out = F.relu(self.bn2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DropoutResNet(MCDropoutModule):
    def __init__(self, block, n_mc_samples, num_blocks, num_classes=10, p_drop=0.2):
        super(MCDropoutModule, self).__init__(n_mc_samples)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv1_drop = ConsistentMCDropout2d(p_drop)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], p_drop, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], p_drop, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], p_drop, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], p_drop, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, p_drop, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, p_drop, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_embeddings=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = (self.linear(out), out) if get_embeddings else self.linear(out)
        return out

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        out = self.conv1(mc_input_BK)
        out = self.conv1_drop(out)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model = model.train()
    model = model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    for X_batch, y_batch in metric_logger.log_every(dataloader, print_freq=print_freq, header=header):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        out = model(X_batch)
        loss = criterion(out, y_batch)
        batch_size = X_batch.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, = generalization.accuracy(out.softmax(dim=-1), y_batch, topk=(1,))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    train_stats = {f"train_{k}": meter.global_avg for k, meter, in metric_logger.meters.items()}
    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, ood_loaders, criterion, device, weights=None):
    test_stats = {}
    model.eval()
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
