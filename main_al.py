import os
import json
import copy
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler

from model import build_model
from utils import seed_everything
from datasets import build_al_dataset
from datasets.data import Data
from metrics.generalization import accuracy
import pylab as plt
from query_stratigies.reweighting_wrapper import LeastConfidenceReweighting_sngp, RandomSamplingReweighting_sngp, \
    QueryByCommitteeReweighting_sngp, BALDReweighting_sngp, BatchBALDReweighting_sngp


# def mesh_plot(args, query_idxs, reweight_model_list, ds_al, train_ds, strategy, title):
#     x, yy = torch.meshgrid(torch.linspace(-4, 4, 51), torch.linspace(-4, 4, 51))
#     zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
#     fig = plt.figure(figsize=(20, 10*len(query_idxs)))
#     count = 0
#     fig.set_tight_layout(True)
#     fig.suptitle(title, fontsize=20, y=1.0)
#
#     for i in range(len(query_idxs)):
#         count += 1
#         mesh_probas_reweighted = reweight_model_list[i].forward_mean_field(mesh_ds.X.to(args.device)).softmax(
#             -1).numpy()
#         # mesh_probas_reweighted = model.forward_mean_field(mesh_ds.X.to(args.device)).softmax(-1).numpy()
#         mesh_utilities = strategy.get_utilities(reweight_model_list[i], mesh_ds)
#         plt.subplot(len(query_idxs), 2, count)
#         plt.title("Utility")
#         s = 50
#         c = plt.contourf(xx, yy, mesh_utilities.reshape(xx.shape), alpha=0.8)
#         plt.colorbar(c)
#         plt.contour(xx, yy, mesh_probas_reweighted[:, 1].reshape(xx.shape), levels=[0.5], colors='black', lw=10)
#         labeled_idx, _ = ds_al.get_labeled_data()
#         unlabeled_idx, _ = ds_al.get_unlabeled_data()
#         plt.scatter(train_ds.X[unlabeled_idx, 0], train_ds.X[unlabeled_idx, 1], c='gray', s=s)
#         plt.scatter(train_ds.X[labeled_idx, 0], train_ds.X[labeled_idx, 1],
#                     c=train_ds.Y[labeled_idx], s=s)
#         plt.scatter(train_ds.X[query_idxs[:i + 1], 0], train_ds.X[query_idxs[:i + 1], 1], c='green', s=100)
#         plt.scatter(train_ds.X[query_idxs[i ], 0], train_ds.X[query_idxs[i], 1], c='red', s=100)
#
#         count+=1
#         plt.subplot(len(query_idxs), 2, count)
#         plt.title("Probability")
#         s = 50
#         c = plt.contourf(xx, yy, mesh_probas_reweighted[:, 1].reshape(xx.shape), alpha=0.8, levels=np.linspace(0, 1, 6))
#         plt.colorbar(c)
#         plt.contour(xx, yy, mesh_probas_reweighted[:, 1].reshape(xx.shape), levels=[0.5], colors='black', lw=10)
#         labeled_idx, _ = ds_al.get_labeled_data()
#         unlabeled_idx, _ = ds_al.get_unlabeled_data()
#         plt.scatter(train_ds.X[unlabeled_idx, 0], train_ds.X[unlabeled_idx, 1], c='gray', s=s)
#         plt.scatter(train_ds.X[labeled_idx, 0], train_ds.X[labeled_idx, 1],
#                     c=train_ds.Y[labeled_idx], s=s)
#         plt.scatter(train_ds.X[query_idxs[:i + 1], 0], train_ds.X[query_idxs[:i + 1], 1], c='green', s=100)
#         plt.scatter(train_ds.X[query_idxs[i], 0], train_ds.X[query_idxs[i], 1], c='red', s=100)
#
#     plt.show()

def mnist_plot(query_idx_dict, dataset, n_round, title, path):
    meta_batch_size = len(query_idx_dict[0])
    fig = plt.figure(figsize=(meta_batch_size, n_round ))
    fig.set_tight_layout(True)
    fig.suptitle(title, fontsize=20, y=1.0)
    count = 0
    for r in range(n_round):
        _, data = dataset.get_add_data(query_idx_dict[r])
        for i in range(meta_batch_size):
            count += 1
            plt.subplot(n_round, meta_batch_size, count)
            plt.title(f"Round {r}", fontsize=5)
            plt.axis('off')
            plt.imshow(data[i][0][0])
            #plt.imshow(dataset[query_idx_dict[r][i]][0][0])
    #plt.show()
    plt.savefig(os.path.join(path, f'{title}.png'))

@hydra.main(version_base=None, config_path="conf", config_name='al_config')
def main(args):
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Get data
    train_ds, test_ds, al_ds, n_classes = build_al_dataset(args)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size)

    # Build model
    model_dict = build_model(args)
    model, train_one_epoch, evaluate = model_dict['model'], model_dict['train_one_epoch'], model_dict['evaluate']
    optimizer, scheduler = model_dict['optimizer'], model_dict['scheduler']
    criterion = nn.CrossEntropyLoss()

    # Save initial states
    init_model_state = copy.deepcopy(model.state_dict())
    init_optimizer_state = copy.deepcopy(optimizer.state_dict())
    init_scheduler_state = copy.deepcopy(scheduler.state_dict())

    # Storage for results
    results = []
    loss_results = []

    # Get initial random samples
    rnd_indices = torch.randperm(len(train_ds))
    init_label_idxs = rnd_indices[:args.init_samples]

    query_idx_dict = dict()

    # Setup active learning components
    ds_al = Data(al_ds, test_ds)
    if args.al_strat == "LeastConfidence":
        strategy = LeastConfidenceReweighting_sngp(ds_al, model, args, enable_reweighting=False)
    elif args.al_strat == "LeastConfidenceReweighted":
        strategy = LeastConfidenceReweighting_sngp(ds_al, model, args, enable_reweighting=True)
    elif args.al_strat == "RandomSampling":
        strategy = RandomSamplingReweighting_sngp(ds_al, model, args, enable_reweighting=False)
    elif args.al_strat == "QueryByCommittee":
        strategy = QueryByCommitteeReweighting_sngp(ds_al, model, args, enable_reweighting=False)
    elif args.al_strat == "QueryByCommitteeReweighted":
        strategy = QueryByCommitteeReweighting_sngp(ds_al, model, args, enable_reweighting=True)
    elif args.al_strat == "BALD":
        strategy = BALDReweighting_sngp(ds_al, model, args, enable_reweighting=False)
    elif args.al_strat == "BALDReweighted":
        strategy = BALDReweighting_sngp(ds_al, model, args, enable_reweighting=True)
    elif args.al_strat == "Batch_BALD":
        strategy = BatchBALDReweighting_sngp(ds_al, model, args, enable_reweighting=False)
    n_round = args.cycle

    strategy.update(init_label_idxs)
    for round in range(n_round):
        model.load_state_dict(init_model_state)
        optimizer.load_state_dict(init_optimizer_state)
        scheduler.load_state_dict(init_scheduler_state)
        label_idx, labeled_ds = ds_al.get_labeled_data()
        train_loader = DataLoader(train_ds, batch_size=args.model.train_batch_size,
                                  sampler=SubsetRandomSampler(label_idx), pin_memory=True)
        for i_epoch in range(args.n_epochs):
            train_stats = train_one_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=args.device, epoch=i_epoch)
            scheduler.step()
            train_stats.update({f"round": round})
            loss_results.append(train_stats)

        model.sample_betas(args.model.draws)
        logits, targets = [], []
        model.eval()
        for inp, tar in test_loader:
            logits.append(model.forward_mean_field(inp.to(args.device)).cpu())
            targets.append(tar)
        logits, targets = torch.cat(logits, dim=0), torch.cat(targets)
        probas = logits.softmax(-1)

        res = {}
        res.update({f"n_train_samples": len(label_idx)})
        res.update({f"test_acc1": accuracy(probas, targets)[0].item()})
        results.append(res)
        print(res)
        query_idxs, reweight_model_list = strategy.query(args.addendum, return_weights=False, subset_size=args.subset)
        print(query_idxs)
        #if n_samples_added == 10:
        #    mesh_plot(model, mesh_ds, args, query_idxs, reweight_model_list, xx, yy, ds_al, train_ds,
        #              strategy, title=f"No Reweight Round:{round} al_strat: {args.al_strat}")
        strategy.update(query_idxs)
        query_idx_dict[round] = query_idxs
    '''
    mnist_plot(query_idx_dict, ds_al, n_round
               , title=f"MetaBatchSize_{args.al_strat}"
               , path = args.output_dir)
    '''
    print(loss_results)
    with open(os.path.join(args.output_dir, 'loss_results.json'), 'w') as f:
        json.dump(loss_results, f)

    print(results)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
