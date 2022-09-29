import numpy as np
from query_stratigies import joint_entropy
import torch
from typing import List


def get_batchbald_batch(
    probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None
) :
    log_probs_N_K_C = torch.log(probs_N_K_C)
    N, K, C = log_probs_N_K_C.shape
    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    conditional_entropies_N = (-probs_N_K_C * torch.log(probs_N_K_C)).sum(2).mean(1)  # compute_conditional_entropy

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in range(batch_size):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return candidate_scores, candidate_indices

