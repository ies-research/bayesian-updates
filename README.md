# SNGP Bayesian Updates
This is the code to reproduce the results of our paper **SNGP with Fast Bayesian Updates for Deep Learning with a Use Case in Active Learning**.

## Install Conda Environment
Make sure you install all requirements using conda and the following line of bash code:
```
conda env create -f environment.yml
```

## Bayesian Updating Experiments
Reproduce experiments for Section 4.
```
# Dataset to use. {MNIST, LETTER, PENDIGITS, FashionMNIST}
DATASET=MNIST

# OOD dataset to use.
DATASET_OOD=[FashionMNIST]

# Random seed for reproducibility.
RANDOM_SEED=1

# Path to save results.
OUTPUT_DIR=./output

# Initial number of samples for training.
N_SAMPLES_START=16

# Maximum number of samples for training.
N_SAMPLES_END=320

# Number of samples which are used for retraining/updating.
N_SAMPLES_ADDED=\[16,32,64\]

# Number of epochs to train the neural network.
N_EPOCHS=200

# Size of the subset used for validation.
VAL_SIZE=10000

# Model config can be found in conf/model {sngp_mnist, sngp_fashion_mnist, sngp_letter, sngp_pendigits}
# Make sure the selected mode is corresponding to $DATASET
MODEL=sngp_mnist

python -u main.py \
        dataset=$DATASET \
        datasets_ood=$DATASET_OOD \
        random_seed=$RANDOM_SEED \
        n_samples_start=$N_SAMPLES_START \
        n_samples_end=$N_SAMPLES_END \
        n_samples_added=$N_SAMPLES_ADDED \
        n_epochs=$N_EPOCHS \
        output_dir=$OUTPUT_DIR \
        val_size=$VAL_SIZE \
        model=$MODEL
```


## Use Case: Active Learning with Bayesian Updates
Reproduce experiments for Section 5.
```
# Number of times to perform updating process.
ADDENDUM=32

# Randomly selected initial labeled data.
INIT_SAMPLES=16

# Active learning cycle, $ADDENDUM samples are selected in each cycle. 
CYCLE=50

# 4 Benchmark Datasets {MNIST, LETTER, PENDIGITS,FashionMNIST}
DATASET=PENDIGITS

# Model config can be found in conf/model {sngp_mnist, sngp_fashion_mnist, sngp_letter, sngp_pendigits}.
# Make sure the selected mode is corresponding to $DATASET.
MODEL=sngp_pendigits

# The following active learning strategies can be selected, corresponding to 
# the baseline and their Bayesian Updates methods mentioned in the paper, respectively.
# {RandomSampling}
# {LeastConfidence, LeastConfidenceReweighted}
# {QueryByCommittee, QueryByCommitteeReweighted}
# {BALD, BALDReweighted, Batch_BALD}
STRATEGY=LeastConfidence

# Random seed for reproducibility.
RANDOM_SEED=0


# Path to save results.
OUTPUT_DIR=./output/

python -u main_al.py \
        dataset=$DATASET \
        model=$MODEL \
        output_dir=$OUTPUT_DIR \
        al_strat=$STRATEGY \
        cycle=$CYCLE \
        init_samples=$INIT_SAMPLES \
        addendum=$ADDENDUM \
        random_seed=$RANDOM_SEED
```


