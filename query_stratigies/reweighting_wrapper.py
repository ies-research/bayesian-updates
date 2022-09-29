import copy
import time
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from query_stratigies.qbc_utility import average_kl_divergence, vote_entropy, compute_vote_vectors
from query_stratigies.sngp_utility import get_gp_features, get_prediction, get_reweighted_model, get_probmatrix
from datetime import datetime
from query_stratigies.batch_bald_utility import get_batchbald_batch

class LeastConfidenceReweighting_sngp():
    def __init__(self, dataset, net, args, enable_reweighting=True):
        self.enable_reweighting = enable_reweighting
        self.args = args
        self.net = net
        self.dataset = dataset

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def get_utilities(self, model, mesh_ds):
        # For 2D visualization only
        mesh_scaled_logits = model.forward_mean_field(mesh_ds.X.to(self.args.device)).softmax(-1)
        probs = mesh_scaled_logits.softmax(-1)
        uncertainties = probs.min(1)[0].numpy()
        return uncertainties

    def query(self, n, return_weights=False, subset_size=None):
        if subset_size!=None and subset_size != 'None':
            unlabeled_idxs, unlabeled_data = self.dataset.get_sampled_unlabeled_data(subset_size)
            print(f"using sampled_unlabeled_data {subset_size}")
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        labeled_mask = np.zeros_like(unlabeled_idxs, dtype=bool)
        self.net.eval()
        queried_sample_idxs = []
        data_loader = DataLoader(unlabeled_data, batch_size=self.args.val_batch_size)
        reweight_model_list = []
        logits_sampled, targets = [], []
        if self.enable_reweighting is False:
            no_rew_start_time = datetime.now()
            for inp, tar in data_loader:
                logits_sampled.append(self.net.forward_mean_field(inp.to(self.args.device)).cpu())
                targets.append(tar)
            logits_sampled, targets = torch.cat(logits_sampled, dim=0), torch.cat(targets)
            probs = logits_sampled.softmax(-1)
            uncertainties = probs.max(1)[0]
            # For 2D visualization only
            if return_weights is True:
                for idx in range(n):
                    reweight_model_list.append(copy.deepcopy(self.net))

            no_rew_end_time = datetime.now()
            print('no_rew Duration: {}'.format(no_rew_end_time - no_rew_start_time))
            return unlabeled_idxs[uncertainties.sort()[1][:n]], reweight_model_list

        if self.enable_reweighting is True:
            rew_start_time = datetime.now()
            for idx in range(n):
                if idx == 0:
                    get_gp_f_start_time = datetime.now()
                    model_reweighted = self.net
                    phis, targets, predictive_covariance = get_gp_features(model_reweighted, data_loader, self.args.device)
                    get_gp_f_end_time = datetime.now()
                    print('get_gp_feature Duration: {}'.format(get_gp_f_end_time - get_gp_f_start_time))
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(model_reweighted)
                else:
                    reweighting_loader = DataLoader(unlabeled_data
                                                    , batch_size=self.args.reweight_batch_size
                                                    , sampler=SubsetRandomSampler(queried_sample_idxs))
                    rew_phis, rew_targets, rew_cov = get_gp_features(self.net, reweighting_loader, self.args.device)
                    model_reweighted = get_reweighted_model(self.net, rew_phis, rew_targets, lmb=self.args.model.lmb)
                    '''
                    model_reweighted = reweight(self.net, reweighting_loader
                                              , device=self.args.device
                                              , lmb=self.args.model.lmb)
                    '''

                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(copy.deepcopy(model_reweighted.to(self.args.device)))

                model_reweighted = model_reweighted.to(self.args.device)
                scaled_logits = get_prediction(model_reweighted, self.args.device, phis, predictive_covariance)
                '''
                logits_sampled, targets = [], []
                for inp, tar in data_loader:
                    logits_sampled.append(model_reweighted.forward_mean_field(inp.to(self.args.device)).cpu())
                    targets.append(tar)
                logits_sampled, targets = torch.cat(logits_sampled, dim=0), torch.cat(targets)
                '''
                # uncer = logits_sampled.softmax(-1).max(1)[0]
                # print(unlabeled_idxs[uncer.sort()[1][:n]])
                probs = scaled_logits.softmax(-1).cpu().numpy()
                uncertainties = probs.max(1)
                uncertainties[labeled_mask] = np.inf
                query_idx = uncertainties.argmin()
                queried_sample_idxs.append(query_idx)
                labeled_mask[query_idx] = True
                # print(unlabeled_idxs[queried_sample_idxs])
            rew_end_time = datetime.now()
            print('rew Duration: {}'.format(rew_end_time - rew_start_time))
            return unlabeled_idxs[queried_sample_idxs], reweight_model_list


class RandomSamplingReweighting_sngp():
    def __init__(self, dataset, net, args, enable_reweighting=True):
        self.enable_reweighting = enable_reweighting
        self.args = args
        self.net = net
        self.dataset = dataset

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def get_utilities(self, model, mesh_ds):
        # For 2D visualization only
        print(mesh_ds.X.shape, len(mesh_ds))
        return np.ones(mesh_ds.X.shape[0])/len(mesh_ds)

    def query(self, n, return_weights=False, subset_size=None):
        if subset_size!=None and subset_size != 'None':
            unlabeled_idxs, unlabeled_data = self.dataset.get_sampled_unlabeled_data(subset_size)
            print(f"using sampled_unlabeled_data {subset_size}")
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        labeled_mask = np.zeros_like(unlabeled_idxs, dtype=bool)
        data_loader = DataLoader(unlabeled_data, batch_size=self.args.val_batch_size)
        self.net.eval()
        queried_sample_idxs = []
        reweight_model_list = []
        if self.enable_reweighting is False:
            queried_sample_idxs = np.random.choice(range(len(unlabeled_data)), n)
            # For 2D visualization only
            if return_weights is True:
                for idx in range(n):
                    reweight_model_list.append(copy.deepcopy(self.net))
            return unlabeled_idxs[queried_sample_idxs], reweight_model_list

        # For 2D visualization only
        if self.enable_reweighting is True:
            for idx in range(n):
                if idx == 0:
                    model_reweighted = self.net
                    phis, targets, predictive_covariance = get_gp_features(model_reweighted, data_loader, self.args.device)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(model_reweighted)
                else:

                    reweighting_loader = DataLoader(unlabeled_data
                                                    , batch_size=self.args.reweight_batch_size
                                                    , sampler=SubsetRandomSampler(queried_sample_idxs))
                    rew_phis, rew_targets, rew_cov = get_gp_features(self.net, reweighting_loader, self.args.device)
                    model_reweighted = get_reweighted_model(self.net, rew_phis, rew_targets, lmb=self.args.model.lmb)
                    '''
                    model_reweighted = reweight(self.net, reweighting_loader
                                              , device=self.args.device
                                              , lmb=self.args.model.lmb)
                    '''
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(copy.deepcopy(model_reweighted.to(self.args.device)))

                query_idx = np.random.choice(range(len(unlabeled_data)))
                queried_sample_idxs.append(query_idx)
                labeled_mask[query_idx] = True
        return unlabeled_idxs[queried_sample_idxs], reweight_model_list


class QueryByCommitteeReweighting_sngp():
    def __init__(self, dataset, net, args, enable_reweighting=True, method="vote_entropy"):
        self.enable_reweighting = enable_reweighting
        self.args = args
        self.net = net
        self.dataset = dataset
        self.method = method

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def get_utilities(self, model, mesh_ds):
        # For 2D visualization only
        logits_sampled = model.forward_sample(mesh_ds.X.to(self.args.device)).cpu()
        probs = logits_sampled.softmax(-1).cpu().numpy()
        if self.method == "KL_divergence":
            utilities_cand = average_kl_divergence(probs)
        elif self.method == "vote_entropy":
            # preds shape (n_estimators, n_samples)
            preds = np.argmax(probs, axis=2)
            try:
                label = self.dataset.train_ds.train_labels
            except:
                label = self.dataset.train_ds.tensors[1].detach().cpu().numpy()
            #votes = compute_vote_vectors(np.array(preds).T, classes=np.unique(label))
            utilities_cand = vote_entropy(np.array(preds).T, np.unique(label))
        return utilities_cand

    def query(self, n, return_weights=False, subset_size=None):
        if subset_size!=None and subset_size != 'None':
            unlabeled_idxs, unlabeled_data = self.dataset.get_sampled_unlabeled_data(subset_size)
            print(f"using sampled_unlabeled_data {subset_size}")
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        labeled_mask = np.zeros_like(unlabeled_idxs, dtype=bool)
        self.net.eval()
        queried_sample_idxs = []
        data_loader = DataLoader(unlabeled_data, batch_size=self.args.val_batch_size)
        reweight_model_list = []
        if self.enable_reweighting is False:
            self.net = self.net
            logits_sampled, targets = [], []
            for inp, tar in data_loader:
                logits_sampled.append(self.net.forward_sample(inp.to(self.args.device)).cpu())
                targets.append(tar)
            logits_sampled, targets = torch.cat(logits_sampled, dim=1), torch.cat(targets)
            # probs shape (n_estimators, n_samples, n_class)
            probs = logits_sampled.softmax(-1).cpu().numpy()
            if self.method == "KL_divergence":
                utilities_cand = average_kl_divergence(probs)
            elif self.method == "vote_entropy":
                # preds shape (n_estimators, n_samples)
                preds = np.argmax(probs, axis=2)
                try:
                    label = self.dataset.train_ds.train_labels
                except:
                    label = self.dataset.train_ds.tensors[1].detach().cpu().numpy()
                #votes = compute_vote_vectors(np.array(preds).T, classes=np.unique(label))
                utilities_cand = vote_entropy(np.array(preds).T, np.unique(label))
            # For 2D visualization only
            if return_weights is True:
                for idx in range(n):
                    reweight_model_list.append(copy.deepcopy(self.net))
            return unlabeled_idxs[utilities_cand.argsort()[-n:]], reweight_model_list

        if self.enable_reweighting is True:
            for idx in range(n):
                if idx == 0:
                    model_reweighted = self.net
                    phis, targets, predictive_covariance = get_gp_features(model_reweighted, data_loader, self.args.device)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(model_reweighted)
                else:

                    reweighting_loader = DataLoader(unlabeled_data
                                                    , batch_size=self.args.reweight_batch_size
                                                    , sampler=SubsetRandomSampler(queried_sample_idxs))
                    rew_phis, rew_targets, rew_cov = get_gp_features(self.net, reweighting_loader, self.args.device)
                    model_reweighted = get_reweighted_model(self.net, rew_phis, rew_targets, lmb=self.args.model.lmb)
                    '''
                    model_reweighted = reweight(self.net, reweighting_loader
                                              , device=self.args.device
                                              , lmb=self.args.model.lmb)
                    '''
                    model_reweighted.sample_betas(self.args.model.draws)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(copy.deepcopy(model_reweighted.to(self.args.device)))

                model_reweighted = model_reweighted.to(self.args.device)
                '''
                logits_sampled, targets = [], []
                for inp, tar in data_loader:
                    logits_sampled.append(model_reweighted.forward_sample(inp.to(self.args.device)).cpu())
                    targets.append(tar)
                logits_sampled, targets = torch.cat(logits_sampled, dim=1), torch.cat(targets)
                '''
                logits_sampled = get_probmatrix(model_reweighted, self.args.device, phis)
                # probs shape (n_estimators, n_samples, n_class)
                probs = logits_sampled.softmax(-1).cpu().numpy()

                if self.method == "KL_divergence":
                    utilities_cand = average_kl_divergence(probs)
                elif self.method == "vote_entropy":
                    # preds shape (n_estimators, n_samples)
                    preds = np.argmax(probs, axis=2)
                    try:
                        label = self.dataset.train_ds.train_labels
                    except:
                        label = self.dataset.train_ds.tensors[1].detach().cpu().numpy()
                    #votes = compute_vote_vectors(np.array(preds).T, classes=np.unique(label))
                    utilities_cand = vote_entropy(np.array(preds).T, np.unique(label))
                utilities_cand[labeled_mask] = -np.inf
                query_idx = utilities_cand.argmax()
                queried_sample_idxs.append(query_idx)
                labeled_mask[query_idx] = True
            return unlabeled_idxs[queried_sample_idxs], reweight_model_list

class BALDReweighting_sngp():
    def __init__(self, dataset, net, args, enable_reweighting=True):
        self.enable_reweighting = enable_reweighting
        self.args = args
        self.net = net
        self.dataset = dataset

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def get_utilities(self, model, mesh_ds):
        # For 2D visualization only
        logits_sampled = model.forward_sample(mesh_ds.X.to(self.args.device)).cpu()
        probs = logits_sampled.softmax(-1).cpu().numpy()
        pb = probs.mean(0)
        entropy1 = (-probs * np.log(pb)).sum(2).mean(0)  #compute_entropy  #(-pb * np.log(pb)).sum(1)
        entropy2 = (-probs * np.log(probs)).sum(2).mean(0) #compute_conditional_entropy
        utilities_cand = entropy1 - entropy2
        return utilities_cand

    def query(self, n, return_weights=False, subset_size=None):
        if subset_size!=None and subset_size != 'None':
            unlabeled_idxs, unlabeled_data = self.dataset.get_sampled_unlabeled_data(subset_size)
            print(f"using sampled_unlabeled_data {subset_size}")
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        labeled_mask = np.zeros_like(unlabeled_idxs, dtype=bool)
        self.net.eval()
        queried_sample_idxs = []
        data_loader = DataLoader(unlabeled_data, batch_size=self.args.val_batch_size)
        reweight_model_list = []
        if self.enable_reweighting is False:
            self.net = self.net
            logits_sampled, targets = [], []
            for inp, tar in data_loader:
                logits_sampled.append(self.net.forward_sample(inp.to(self.args.device)).cpu())
                targets.append(tar)
            logits_sampled, targets = torch.cat(logits_sampled, dim=1), torch.cat(targets)
            # probs shape (n_estimators, n_samples, n_class)
            probs = logits_sampled.softmax(-1).cpu().numpy()
            pb = probs.mean(0)
            entropy1 = (-pb * np.log(pb)).sum(1)
            entropy2 = (-probs * np.log(probs)).sum(2).mean(0)
            utilities_cand =  entropy1 - entropy2
            # For 2D visualization only
            if return_weights is True:
                for idx in range(n):
                    reweight_model_list.append(copy.deepcopy(self.net))
            return unlabeled_idxs[utilities_cand.argsort()[-n:]], reweight_model_list

        if self.enable_reweighting is True:
            for idx in range(n):
                if idx == 0:
                    model_reweighted = self.net
                    phis, targets, predictive_covariance = get_gp_features(model_reweighted, data_loader, self.args.device)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(model_reweighted)
                else:

                    reweighting_loader = DataLoader(unlabeled_data
                                                    , batch_size=self.args.reweight_batch_size
                                                    , sampler=SubsetRandomSampler(queried_sample_idxs))
                    rew_phis, rew_targets, rew_cov = get_gp_features(self.net, reweighting_loader, self.args.device)
                    model_reweighted = get_reweighted_model(self.net, rew_phis, rew_targets, lmb=self.args.model.lmb)
                    '''
                    model_reweighted = reweight(self.net, reweighting_loader
                                              , device=self.args.device
                                              , lmb=self.args.model.lmb)
                    '''
                    model_reweighted.sample_betas(self.args.model.draws)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(copy.deepcopy(model_reweighted.to(self.args.device)))

                model_reweighted = model_reweighted.to(self.args.device)

                logits_sampled = get_probmatrix(model_reweighted, self.args.device, phis)
                # probs shape (n_estimators, n_samples, n_class)
                probs = logits_sampled.softmax(-1).cpu().numpy()
                pb = probs.mean(0)
                entropy1 = (-pb * np.log(pb)).sum(1)  #compute_entropy
                entropy2 = (-probs * np.log(probs)).sum(2).mean(0) #compute_conditional_entropy
                utilities_cand = entropy1 - entropy2

                utilities_cand[labeled_mask] = -np.inf
                query_idx = utilities_cand.argmax()
                queried_sample_idxs.append(query_idx)
                labeled_mask[query_idx] = True
            return unlabeled_idxs[queried_sample_idxs], reweight_model_list


class BatchBALDReweighting_sngp():
    def __init__(self, dataset, net, args, enable_reweighting=True):
        self.enable_reweighting = enable_reweighting
        self.args = args
        self.net = net
        self.dataset = dataset

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def get_utilities(self, model, mesh_ds):
        # For 2D visualization only
        logits_sampled = model.forward_sample(mesh_ds.X.to(self.args.device)).cpu()
        probs = logits_sampled.softmax(-1).cpu().numpy()
        pb = probs.mean(0)
        entropy1 = (-pb * np.log(pb)).sum(1)
        entropy2 = (-probs * np.log(probs)).sum(2).mean(0)
        utilities_cand = entropy1 - entropy2
        return utilities_cand

    def query(self, n, return_weights=False, subset_size=None):
        if subset_size!=None and subset_size != 'None':
            unlabeled_idxs, unlabeled_data = self.dataset.get_sampled_unlabeled_data(subset_size)
            print(f"using sampled_unlabeled_data {subset_size}")
        else:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        labeled_mask = np.zeros_like(unlabeled_idxs, dtype=bool)
        self.net.eval()
        queried_sample_idxs = []
        data_loader = DataLoader(unlabeled_data, batch_size=self.args.val_batch_size)
        reweight_model_list = []
        if self.enable_reweighting is False:
            self.net = self.net
            logits_sampled, targets = [], []
            for inp, tar in data_loader:
                logits_sampled.append(self.net.forward_sample(inp.to(self.args.device)).cpu())
                targets.append(tar)
            logits_sampled, targets = torch.cat(logits_sampled, dim=1), torch.cat(targets)
            # probs shape (n_estimators, n_samples, n_class)
            probs = logits_sampled.softmax(-1).cpu().numpy()
            probs_N_K_C = torch.tensor(np.transpose(probs, (1,0,2)))
            print(probs_N_K_C.shape)
            candidate_scores, candidate_indices = get_batchbald_batch(probs_N_K_C, n,  probs_N_K_C.shape[0])

            # For 2D visualization only
            if return_weights is True:
                for idx in range(n):
                    reweight_model_list.append(copy.deepcopy(self.net))
            return unlabeled_idxs[candidate_indices], reweight_model_list

        if self.enable_reweighting is True:
            for idx in range(n):
                if idx == 0:
                    model_reweighted = self.net
                    phis, targets, predictive_covariance = get_gp_features(model_reweighted, data_loader, self.args.device)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(model_reweighted)
                else:

                    reweighting_loader = DataLoader(unlabeled_data
                                                    , batch_size=self.args.reweight_batch_size
                                                    , sampler=SubsetRandomSampler(queried_sample_idxs))
                    rew_phis, rew_targets, rew_cov = get_gp_features(self.net, reweighting_loader, self.args.device)
                    model_reweighted = get_reweighted_model(self.net, rew_phis, rew_targets, lmb=self.args.model.lmb)
                    '''
                    model_reweighted = reweight(self.net, reweighting_loader
                                              , device=self.args.device
                                              , lmb=self.args.model.lmb)
                    '''
                    model_reweighted.sample_betas(self.args.model.draws)
                    # For 2D visualization only
                    if return_weights is True:
                        reweight_model_list.append(copy.deepcopy(model_reweighted.to(self.args.device)))

                model_reweighted = model_reweighted.to(self.args.device)

                logits_sampled = get_probmatrix(model_reweighted, self.args.device, phis)
                # probs shape (n_estimators, n_samples, n_class)
                probs = logits_sampled.softmax(-1).cpu().numpy()
                probs_N_K_C = torch.tensor(np.transpose(probs, (1, 0, 2)))
                candidate_scores, candidate_indices = get_batchbald_batch(probs_N_K_C, 1, probs_N_K_C.shape[0])
                query_idx = candidate_indices[0]
                queried_sample_idxs.append(query_idx)
                labeled_mask[query_idx] = True
            return unlabeled_idxs[queried_sample_idxs], reweight_model_list