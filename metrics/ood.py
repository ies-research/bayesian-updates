import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def entropy_fn(probas):
    probas = probas.clone()
    eps = torch.finfo(probas.dtype).eps
    probas = probas.clamp(min=eps, max=(1-eps))
    return - torch.sum(probas * probas.log(), -1)


def ood_auroc(score_id: torch.Tensor, score_ood: torch.Tensor):
    """Computes the AUROC and assumes that a higher score is OOD.

    Args:
        score_id (torch.Tensor): Score of in-distribution samples.
        score_ood (torch.Tensor): Score of out-of-distribution samples.

    Returns:
        float: The AUROC score.
    """
    y_true = torch.cat((torch.zeros(len(score_id)), torch.ones(len(score_ood))))
    y_score = torch.cat((score_id, score_ood))
    return roc_auc_score(y_true, y_score)


def ood_aupr(score_id: torch.Tensor, score_ood: torch.Tensor):
    """Computes the AUROC and assumes that a higher score is OOD.

    Args:
        score_id (torch.Tensor): Score of in-distribution samples.
        score_ood (torch.Tensor): Score of out-of-distribution samples.

    Returns:
        float: The AUROC score.
    """
    y_true = torch.cat((torch.zeros(len(score_id)), torch.ones(len(score_ood))))
    y_score = torch.cat((score_id, score_ood))
    return average_precision_score(y_true, y_score)
