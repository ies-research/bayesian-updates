import json

import numpy as np
import pylab as plt

from omegaconf import OmegaConf
from pathlib import Path
from scipy.integrate import simpson
from numpy import trapz
from sklearn.metrics import auc
import matplotlib as mpl
from scipy.stats import ttest_rel

def load_results(exp_path: str):
    exp_path = Path(exp_path)
    json_file = exp_path / 'results.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    config_file = exp_path / '.hydra' / 'config.yaml'
    with open(config_file, 'r') as f:
        args = OmegaConf.load(f)
    return data, args

def load_experiments(root_path, folder_name):
    experiment_paths = sorted(list(root_path.glob(folder_name)))
    # experiment_paths = filter(lambda x: 'batch' not in str(x), experiment_paths)

    # Load results of experiments
    experiments = {}
    for exp_path in experiment_paths:
        results, args = load_results(exp_path)
        experiments[exp_path.stem] = {'results': results, 'args': args}

    print(experiments.keys())
    num_experiments = len(experiments.keys())
    return experiments, num_experiments

def load_results_mean(experiments):
    num_experiments = len(experiments.keys())
    all_results = []
    for exp_name, exp_data in experiments.items():
        result = dict()
        res = exp_data['results']
        result['n_train_samples'] = []
        result['test_acc1'] = []
        for value in res:
            result['n_train_samples'].append(value['n_train_samples'])
            result['test_acc1'].append(value['test_acc1'])
        all_results.append(result)

    result_mean = dict()
    for n_samples_added, value in all_results[0].items():
        for exp_n in range(num_experiments):
            if exp_n == 0:
                result_mean = dict()
                result_mean['n_train_samples'] = np.array(
                    all_results[exp_n]['n_train_samples'])
                result_mean['test_acc1'] = np.array(
                    all_results[exp_n]['test_acc1'])
            else:
                result_mean['n_train_samples'] += all_results[exp_n][
                    'n_train_samples']
                result_mean['test_acc1'] += np.array(
                    all_results[exp_n]['test_acc1'])

        result_mean['n_train_samples'] = result_mean['n_train_samples'].astype('float64')
        result_mean['n_train_samples'] /= num_experiments
        result_mean['test_acc1'] /= num_experiments
    return result_mean, all_results


"""Evaluate server experiments with different random seeds."""
root_path = Path('path/to/experiments')  #
# MNIST LETTER PENDIGITS FashionMNIST
DATA="PENDIGITS"

folder_name = f'{DATA}_LeastConfidence_*'
ep_US_no_rew, num_ep_US_no_rew = load_experiments(root_path, folder_name)
result_mean_US_no_rew, all_results_US_no_rew = load_results_mean(ep_US_no_rew)

folder_name = f'{DATA}_LeastConfidenceReweighted_*'
ep_US_rew, num_ep_US_rew = load_experiments(root_path, folder_name)
result_mean_US_rew, all_results_US_rew = load_results_mean(ep_US_rew)

folder_name = f'{DATA}_RandomSampling_*'
experiments_RS, num_experiments_RS = load_experiments(root_path, folder_name)
result_mean_RS, all_results_RS = load_results_mean(experiments_RS)

folder_name = f'{DATA}_BALD_*'
ep_BALD_no_rew, num_ep_BALD_no_rew = load_experiments(root_path, folder_name)
result_mean_BALD_no_rew, all_results_BALD_no_rew = load_results_mean(ep_BALD_no_rew)

folder_name = f'{DATA}_BALDReweighted_*'
ep_BALD_rew, num_ep_BALD_rew = load_experiments(root_path, folder_name)
result_mean_BALD_rew, all_results_BALD_rew = load_results_mean(ep_BALD_rew)

folder_name = f'{DATA}_QueryByCommittee_*'
ep_QBC_no_rew, num_ep_QBC_no_rew = load_experiments(root_path, folder_name)
result_mean_QBC_no_rew, all_results_QBC_no_rew = load_results_mean(ep_QBC_no_rew)

folder_name = f'{DATA}_QueryByCommitteeReweighted_*'
epQBC_rew, num_ep_QBC_rew = load_experiments(root_path, folder_name)
result_mean_QBC_rew, all_results_QBC_rew = load_results_mean(epQBC_rew)

folder_name = f'{DATA}_Batch_BALD_*'
experiments_Batch_BALD, num_experiments_Batch_BALD = load_experiments(root_path, folder_name)
result_mean_Batch_BALD, all_results_Batch_BALD = load_results_mean(experiments_Batch_BALD)

num_experiments = num_ep_US_no_rew

count=0
plt.figure(figsize=(7, 5))
plt.title("Different Strategies with different Meta Batch Size")
n_samples_added=32
if DATA=="PENDIGITS":
    plt.title("PDIGITS")
elif DATA=="FashionMNIST":
    plt.title("FMNIST")
else:
   plt.title(f"{DATA}")
mpl.style.use('seaborn')
'''
for exp_n in range(num_experiments):

    plt.plot(all_results_RS[exp_n]['n_train_samples']
             , all_results_RS[exp_n]['test_acc1_no_reweight']
             , '-', color=f'C{2 + 1}', alpha=0.05)
    plt.plot(all_results_US[exp_n]['n_train_samples']
             , all_results_US[exp_n]['test_acc1_no_reweight']
             , '-', color=f'C{2 + 2}', alpha=0.05)
    plt.plot(all_results_US[exp_n]['n_train_samples']
             , all_results_US[exp_n]['test_acc1_reweight']
             , '--', color=f'C{2 + 2}', alpha=0.05)
    plt.plot(all_results_QBC[exp_n]['n_train_samples']
             , all_results_QBC[exp_n]['test_acc1_no_reweight']
             , '-', color=f'C{2 + 3}', alpha=0.05)
    plt.plot(all_results_QBC[exp_n]['n_train_samples']
             , all_results_QBC[exp_n]['test_acc1_reweight']
             , '--', color=f'C{2 + 3}', alpha=0.05)
    plt.plot(all_results_BALD[exp_n]['n_train_samples']
             , all_results_BALD[exp_n]['test_acc1_no_reweight']
             , '-', color=f'C{2 + 4}', alpha=0.05)
    plt.plot(all_results_BALD[exp_n]['n_train_samples']
             , all_results_BALD[exp_n]['test_acc1_reweight']
             , '--', color=f'C{2 + 4}', alpha=0.05)
    plt.plot(all_results_Batch_BALD[exp_n]['n_train_samples']
             , all_results_BALD[exp_n]['test_acc1_no_reweight']
             , '-', color=f'C{2 + 5}', alpha=0.05)

    plt.plot(all_results_Batch_BALD[exp_n]['n_train_samples']
             , all_results_Batch_BALD[exp_n]['test_acc1']
             , '--', color=f'C{2 + 5}', alpha=0.05)
    '''
rs = np.array([i["test_acc1"] for i in all_results_RS])
us = np.array([i["test_acc1"] for i in all_results_US_no_rew])
us_rew = np.array([i["test_acc1"] for i in all_results_US_rew])
qbc = np.array([i["test_acc1"] for i in all_results_QBC_no_rew])
qbc_rew = np.array([i["test_acc1"] for i in all_results_QBC_rew])
batch_bald = np.array([i["test_acc1"] for i in all_results_Batch_BALD])
bald_rew = np.array([i["test_acc1"] for i in all_results_BALD_rew])
bald = np.array([i["test_acc1"] for i in all_results_BALD_no_rew])

plt.plot(result_mean_RS['n_train_samples']
         , result_mean_RS['test_acc1']/100.0
         , '--', color=f'k', label=f'RAND')
#plt.grid(True)
#print(auc(np.arange (0, 1, 1/len(result_mean_RS['test_acc1'])), result_mean_RS['test_acc1']))
#area = trapz(result_mean_RS['test_acc1'], dx=1/len(result_mean_RS['test_acc1']))
print("RS area =", rs.mean())
plt.plot(result_mean_RS['n_train_samples'], result_mean_US_no_rew['test_acc1']/100.0
         , '-', linewidth=1.5, color=f'g', label=f'US')
plt.plot(result_mean_RS['n_train_samples'], result_mean_US_rew['test_acc1']/100.0
         , '-', marker='o', markersize=5, linewidth=1.5, color=f'g', label=f'US (update)')

print("US area =", us.mean())
print("US Rew area =", us_rew.mean())

plt.plot(result_mean_RS['n_train_samples'], result_mean_BALD_no_rew['test_acc1']/100.0
         , '-', color=f'r', linewidth=1.5, label=f'BALD')

plt.plot(result_mean_RS['n_train_samples'], result_mean_BALD_rew['test_acc1']/100.0
         , '-', marker='o', markersize=5, linewidth=1.5, color=f'r', label=f'BALD (update)')

print("BALD area =", bald.mean())
print("BALD Rew area =", bald_rew.mean())

plt.plot(result_mean_RS['n_train_samples'], result_mean_QBC_no_rew['test_acc1']/100.0
         , '-', linewidth=1.5, color=f'b', label=f'QBC')
plt.plot(result_mean_RS['n_train_samples'], result_mean_QBC_rew['test_acc1']/100.0
         , '-', marker='o', markersize=5, linewidth=1.5,color=f'b', label=f'QBC (update)')

print("QBC area =", qbc.mean())
print("QBC Rew area =", qbc_rew.mean())

plt.plot(result_mean_RS['n_train_samples'], result_mean_Batch_BALD['test_acc1']/100.0
         , '-', linewidth=1.5, color=f'm', label=f'BatchBALD')
print("Batch BALD area =", batch_bald.mean())
plt.ylabel("ACC")
plt.xlabel("# samples in $\mathcal{D}$")
#plt.ylim(0.85, 1.0)
plt.ylim(0.6)
plt.tight_layout()
plt.legend(fontsize=12)
plt.show()

print("US", ttest_rel(us.mean(axis=1), us_rew.mean(axis=1), alternative="less"))
print("QBC", ttest_rel(qbc.mean(axis=1), qbc_rew.mean(axis=1), alternative="less"))
print("bald ub", ttest_rel(bald_rew.mean(axis=1), batch_bald.mean(axis=1), alternative="less"))
print("bald u", ttest_rel(bald.mean(axis=1), bald_rew.mean(axis=1), alternative="less"))
print("bald b", ttest_rel(bald.mean(axis=1), batch_bald.mean(axis=1), alternative="less"))