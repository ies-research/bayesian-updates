import os
import time
import json
import copy
import hydra

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import build_model
from utils import seed_everything
from datasets import build_dataset, build_ood_dataset

@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args):
    print(args)
    seed_everything(args.random_seed)
    print("Start experimenting.")

    train_ds, test_ds, n_classes = build_dataset(args)
    val_sampler = None if args.val_size == -1 else range(args.val_size)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, sampler=val_sampler)
    ood_datasets = build_ood_dataset(args)
    ood_loaders = {name: DataLoader(ds, args.val_batch_size, sampler=val_sampler) for name, ds in ood_datasets.items()}

    # build model
    model_dict = build_model(args)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, scheduler = model_dict['optimizer'], model_dict['scheduler']
    reweight = model_dict['reweight']
    criterion = nn.CrossEntropyLoss()

    # Save initial states
    init_model_state = copy.deepcopy(model.state_dict())
    init_optimizer_state = copy.deepcopy(optimizer.state_dict())
    init_scheduler_state = copy.deepcopy(scheduler.state_dict())

    results = []
    subset_sizes = torch.arange(
        args.n_samples_start,
        args.n_samples_end+1+max(args.n_samples_added),
        min(args.n_samples_added)
    )
    n_subsets = len(subset_sizes)

    rnd_indices = torch.randperm(len(train_ds))
    for i_subset, subset_size in enumerate(subset_sizes):
        print(f'[{i_subset}/{n_subsets}] Subset Size: {subset_size}')

        # Baseline
        print('Training Baseline Model')
        # seed_everything(args.random_seed)
        model.load_state_dict(init_model_state)
        optimizer.load_state_dict(init_optimizer_state)
        scheduler.load_state_dict(init_scheduler_state)

        indices = rnd_indices[:subset_size]
        # Drop last to avoid having small batches which make training bad
        train_loader = DataLoader(train_ds, batch_size=args.model.train_batch_size,
                                  sampler=SubsetRandomSampler(indices=indices),
                                  drop_last=(len(indices) > args.model.train_batch_size))
        t1 = time.time()
        for i_epoch in range(args.n_epochs):
            train_one_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=args.device, epoch=i_epoch)
            scheduler.step()
        training_time = time.time() - t1
        test_stats_baseline = evaluate(model, test_loader, ood_loaders=ood_loaders,
                                       criterion=criterion, device=args.device)
        test_stats_baseline['training_time'] = training_time

        # Reweighting
        if args.model.name == 'mc_dropout':
            # resets dropout mask
            model.train()
            model.eval()

        res = {}
        res['n_train_samples'] = subset_size.item()
        res['baseline_test_stats'] = test_stats_baseline
        for n_samples_added in args.n_samples_added:
            print(f'Reweighting with {n_samples_added} samples...')
            indices = rnd_indices[subset_size:subset_size+n_samples_added]
            reweighting_loader = DataLoader(train_ds, batch_size=args.reweight_batch_size,
                                            sampler=SubsetRandomSampler(indices))

            # Return reweighted model?
            print('Evaluating...')
            if args.model.name == 'sngp':
                t1 = time.time()
                model_reweighted = reweight(model, reweighting_loader, device=args.device, lmb=args.model.lmb)
                training_time = time.time() - t1
                test_stats_reweighted = evaluate(model_reweighted, test_loader, ood_loaders=ood_loaders,
                                                 criterion=criterion, device=args.device)
                test_stats_reweighted['training_time'] = training_time
            else:
                if args.model.name == "sngp_sampling":
                    t1 = time.time()
                    weights = reweight(model, reweighting_loader, device=args.device, n_draws=args.model.draws)
                    training_time = time.time() - t1
                    test_stats_reweighted = evaluate(model, test_loader, ood_loaders=ood_loaders,  criterion=criterion, device=args.device, n_draws=args.model.draws, weights=weights, sample_betas=False)
                else:
                    t1 = time.time()
                    weights = reweight(model, reweighting_loader, device=args.device)
                    training_time = time.time() - t1
                    test_stats_reweighted = evaluate(model, test_loader, ood_loaders=ood_loaders,  criterion=criterion, device=args.device, weights=weights)
                test_stats_reweighted['training_time'] = training_time

            res[f"reweighted{n_samples_added}_test_stats"] = test_stats_reweighted

        print(res)
        results.append(res)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
