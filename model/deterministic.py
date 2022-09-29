import torch
import torch.nn as nn

from metrics import generalization, calibration
from utils import MetricLogger, SmoothedValue


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, print_freq=200):
    model.train()
    model.to(device)

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch [{epoch}]" if epoch is not None else "  Train: "

    # Train the epoch
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

    # Forward prop in distribution
    logits_id, targets_id, = [], []
    for inputs, targets in dataloader_id:
        inputs, targets = inputs.to(device), targets.to(device)
        logits_id.append(model(inputs))
        targets_id.append(targets)
    logits_id = torch.cat(logits_id, dim=0).cpu()
    targets_id = torch.cat(targets_id, dim=0).cpu()

    # Test Loss and Accuracy for in domain testset
    acc1 = generalization.accuracy(logits_id, targets_id, (1,))[0]
    loss = criterion(logits_id, targets_id)
    test_stats.update({
        'loss': loss.item(),
        'acc1': acc1.item(),
    })

    # Calibration
    criterion_tce = calibration.TopLabelCalibrationError()
    criterion_mce = calibration.MarginalCalibrationError()
    criterion_bs = calibration.BrierScore()
    criterion_nll = nn.CrossEntropyLoss()

    probas_id = logits_id.softmax(-1)
    nll = criterion_nll(logits_id, targets_id)
    brier_score = criterion_bs.forward(probas_id, targets_id)
    tce = criterion_tce(probas_id, targets_id)
    mce = criterion_mce(probas_id, targets_id)

    test_stats.update({
        'brier_score': brier_score.item(),
        'nll': nll.item(),
        'tce': tce.item(),
        'mce': mce.item(),
        # 'tcp': tcp,
        # 'mcp': mcp,
    })

    for ds_name, ood_loader in ood_loaders.items():
        pass

    test_stats = {f"test_{k}": v for k, v in test_stats.items()}
    return test_stats
