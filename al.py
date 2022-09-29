# Python
import os
import copy
import math
import random

# Torch
import hydra
import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from model import build_model
from model.sngp import reweight
from datasets import build_al_dataset
from metrics.ood import entropy_fn


@hydra.main(version_base=None, config_path="conf", config_name='al_config')
def main(args):
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir)

    INIT_SAMPLES = args.init_samples
    ADDENDUM = args.addendum
    CYCLES = args.cycle
    SUBSET = args.subset
    STRAT = args.al_strat

    # Set random stuff
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    train_ds, test_ds, al_ds, n_classes = build_al_dataset(args)

    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
    NUM_TRAIN = len(train_ds)
    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:INIT_SAMPLES]
    unlabeled_set = indices[INIT_SAMPLES:]

    # Note that one could use drop_last when training batch size and addendum are different
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, sampler=range(args.val_size))

    # Model
    model_dict = build_model(args)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer = model_dict['optimizer']
    criterion = nn.CrossEntropyLoss()

    init_model_state = copy.deepcopy(model.state_dict())
    init_optimizer_state = copy.deepcopy(optimizer.state_dict())

    results = []
    # Active learning cycles
    for i_cycle in range(CYCLES):
        results_cycle = {}

        # Loss, criterion and scheduler (re)initialization
        if args.cold_start:
            model.load_state_dict(init_model_state)
        optimizer.load_state_dict(init_optimizer_state)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

        # Training and test
        # Create a new dataloader for the labeled dataset
        train_loader = DataLoader(train_ds, batch_size=args.model.train_batch_size,
                                  sampler=SubsetRandomSampler(labeled_set), pin_memory=True,
                                  drop_last=len(labeled_set) > args.model.train_batch_size)
        train_history = []
        for i_epoch in range(args.n_epochs):
            train_stats = train_one_epoch(model, train_loader, criterion=criterion,
                                          optimizer=optimizer, device=args.device, epoch=i_epoch)
            scheduler.step()
            train_history.append(train_stats)
            for key, val in train_stats.items():
                writer.add_scalar(tag=f'cycle={i_cycle}/{key}', scalar_value=val, global_step=i_epoch)
        test_stats = evaluate(model, test_loader, ood_loaders={}, criterion=criterion, device=args.device)
        for key, val in test_stats.items():
            try:
                writer.add_scalar(tag=f'test/{key}', scalar_value=val, global_step=i_cycle)
            except:
                print(f'Could not write {key} to summary writer.')

        print('Cycle {}/{} || Label set size {}: Test acc {}'.format(i_cycle +
              1, CYCLES, len(labeled_set), test_stats['test_acc1']))

        # Logging
        results_cycle['n_labeled_samples'] = len(labeled_set)
        results_cycle['train_history'] = train_history
        results_cycle['test_stats'] = test_stats
        results.append(results_cycle)

        if i_cycle+1 == CYCLES:
            break

        # Randomly sample 10000 unlabeled data points
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:SUBSET]

        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(al_ds, batch_size=args.val_batch_size, sampler=subset)

        # Measure uncertainty of each data points in the subset
        if STRAT == 'random':
            remaining_unlabeled_set, new_labeled_set = random_sampling(
                ADDENDUM, subset, unlabeled_loader
            )
        elif STRAT == 'uncertainty':
            remaining_unlabeled_set, new_labeled_set = uncertainty_sampling(
                model, ADDENDUM, subset, unlabeled_loader, args.device
            )
        elif STRAT == "reweighted-uncertainty":
            remaining_unlabeled_set, new_labeled_set = reweighted_sampling(
                uncertainty_sampling, model, ADDENDUM, subset, unlabeled_loader, al_ds, args.val_batch_size, lmb=args.model.lmb, device=args.device
            )
        elif STRAT == "bald":
            remaining_unlabeled_set, new_labeled_set = bald_sampling(
                model, ADDENDUM, subset, unlabeled_loader, args.device, 100
            )
        elif STRAT == "reweighted-bald":
            remaining_unlabeled_set, new_labeled_set = reweighted_sampling(
                bald_sampling, model, ADDENDUM, subset,
                unlabeled_loader, al_ds, args.val_batch_size,
                lmb=args.model.lmb, device=args.device, strategy_dict={"n_draws": 100}
            )
        else:
            raise ValueError("Invalid active learning strategy.")

        # Update the labeled dataset and the unlabeled dataset, respectively
        labeled_set += new_labeled_set
        unlabeled_set = remaining_unlabeled_set + unlabeled_set[SUBSET:]

    path = os.path.join(args.output_dir, 'results.pth')
    print(f'Saving results to {path}')
    torch.save(results, path)


@torch.no_grad()
def random_sampling(addendum, subset, unlabeled_loader):
    # Compute utilities.
    n_samples = sum([len(inputs) for inputs, _ in unlabeled_loader])
    utilities = torch.rand(n_samples)

    # Index in ascending order
    asc_sorted_idx = np.argsort(utilities)

    # Update labeled and unlabeled set.
    new_labeled_set = torch.tensor(subset)[asc_sorted_idx][-addendum:].tolist()
    remaining_unlabeled_set = torch.tensor(subset)[asc_sorted_idx][:-addendum].tolist()

    return remaining_unlabeled_set, new_labeled_set


@torch.no_grad()
def uncertainty_sampling(model, addendum, subset, unlabeled_loader, device):
    # Compute utilities.
    model.eval()
    model.to(device)
    utilities = []
    for (inputs, labels) in unlabeled_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model.forward_mean_field(inputs)
        probas = logits.softmax(-1)
        entropies = entropy_fn(probas)
        utilities.append(entropies)
    utilities = torch.cat(utilities).cpu()

    # Index in ascending order
    asc_sorted_idx = np.argsort(utilities)

    # Update labeled and unlabeled set.
    new_labeled_set = torch.tensor(subset)[asc_sorted_idx][-addendum:].tolist()
    remaining_unlabeled_set = torch.tensor(subset)[asc_sorted_idx][:-addendum].tolist()

    return remaining_unlabeled_set, new_labeled_set


def reweighted_sampling(strategy, model, addendum, subset, unlabeled_loader, al_ds, batch_size, lmb, device, strategy_dict=None):
    if strategy_dict is None:
        strategy_dict = {}
    new_labeled_set = []
    remaining_unlabeled_set = subset
    labeled_loader = None
    for i in range(addendum):
        model.eval()
        if labeled_loader is not None:
            # Reweight model on new data.
            model = reweight(
                model, labeled_loader, device, lmb=lmb
            )
        remaining_unlabeled_set, labeled_set = strategy(
            model, 1, remaining_unlabeled_set, unlabeled_loader, device, **strategy_dict
        )
        new_labeled_set += labeled_set
        unlabeled_loader = DataLoader(al_ds, batch_size=batch_size, sampler=remaining_unlabeled_set)
        labeled_loader = DataLoader(al_ds, sampler=labeled_set)
    return remaining_unlabeled_set, new_labeled_set


def bald_sampling(model, addendum, subset, unlabeled_loader, device, n_draws):
    # Compute utilities.
    model.eval()
    model.to(device)
    log_probas = []
    model.sample_betas(n_draws)
    for (inputs, labels) in unlabeled_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model.forward_sample(inputs).permute(1, 0, 2)
        log_probas.append(logits.log_softmax(dim=-1))
    log_probas = torch.cat(log_probas).cpu()
    utilities = -compute_conditional_entropy(log_probas)
    utilities += compute_entropy(log_probas)

    # Index in ascending order
    asc_sorted_idx = np.argsort(utilities)

    # Update labeled and unlabeled set.
    new_labeled_set = torch.tensor(subset)[asc_sorted_idx][-addendum:].tolist()
    remaining_unlabeled_set = torch.tensor(subset)[asc_sorted_idx][:-addendum].tolist()

    return remaining_unlabeled_set, new_labeled_set


def compute_conditional_entropy(log_probas):
    n_members = log_probas.shape[1]
    nats = log_probas * torch.exp(log_probas)
    entropies = - torch.sum(nats, dim=(1, 2)) / n_members
    return entropies


def compute_entropy(log_probas):
    n_members = log_probas.shape[1]
    mean_log_probas = torch.logsumexp(log_probas, dim=1) - math.log(n_members)
    nats = mean_log_probas * torch.exp(mean_log_probas)
    entropies = -torch.sum(nats, dim=1)
    return entropies


if __name__ == '__main__':
    main()
