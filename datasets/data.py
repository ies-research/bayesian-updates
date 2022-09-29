import numpy as np
import torch
from torchvision import datasets
from random import sample
from torch.utils.data import Subset
class Data:
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
        
        self.n_pool = len(train_ds)
        self.n_test = len(test_ds)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, Subset(self.train_ds, labeled_idxs)

    def get_add_data(self, add_data_index):
        return add_data_index, Subset(self.train_ds, add_data_index)

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, Subset(self.train_ds, unlabeled_idxs)

    def get_sampled_unlabeled_data(self, subset_size):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        sampled_idxs = np.random.choice(range(len(unlabeled_idxs)), size=subset_size, replace=False)
        sampled_unlabeled_idxs = unlabeled_idxs[sampled_idxs]
        return sampled_unlabeled_idxs, Subset(self.train_ds, sampled_unlabeled_idxs)

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.train_ds

    def get_test_data(self):
        return self.test_ds
    

