import argparse
from random import seed
from mylib.utils import fix_seed
from mylib.data.data_loader import load_ucidata
import collections
import numpy as np
from run_dnl import run_dnl
import tools
import pandas as pd
from kmeans import run_kmeans
import os
import argparse
import scipy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections

import cloth1m_labels

# adjust the load data function to load training data only
from mylib.data.data_loader.load_ucidata import load_ucidata2

from mylib.data.dataset.util import noisify_multiclass_symmetric

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from cal_acc import *
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def _hungarian_match(flat_preds, flat_targets):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]
    v, counts = np.unique(flat_preds, return_counts=True)

    num_k = len(v)
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    row, col = linear_sum_assignment(num_samples - num_correct)
    return row, col


def get_prime_Y(noisy_classes, pred_classes, mappings):
    prime_Y = np.zeros(len(noisy_classes))
    for i in range(len(pred_classes)):
        prime_Y[i] = mappings[pred_classes[i]]

    return prime_Y


def count_m(noisy_Y, prime_Y):
    values, counts = np.unique(prime_Y, return_counts=True)
    # print(values)
    length = len(values)
    m = np.zeros((length, length))
    # print(counts)

    for i in range(noisy_Y.shape[0]):
        m[int(prime_Y[i])][int(noisy_Y[i])] += 1

    sum_matrix = np.tile(counts, (len(values), 1)).transpose()
    # print(sum_matrix)
    # print(m/sum_matrix)
    return m/sum_matrix

# define K-means clustering algorithm


def run_kmeans2(dataset):

    X = dataset.data
    clean_Y = dataset.clean_targets
    tilde_Y = dataset.targets
    values, counts = np.unique(clean_Y, return_counts=True)

    kmeans = KMeans(n_clusters=len(values))
    kmeans.fit(X, tilde_Y)
    identified_clusters = kmeans.fit_predict(X)

    # note that to better match the cluster Id to tilde_Y,
    # we could use hat_clean Y which obtained by current noise-robust method, but for the simple dataset , it may not necessary

    idx2 = _hungarian_match(identified_clusters, tilde_Y)
    prime_Y = get_prime_Y(tilde_Y, identified_clusters, idx2[1])
    # yz: directly return prime_Y without using count_m
    return prime_Y


# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE")
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--z_dim', type=int, default=25,
                    help='dimension of hidden variable Z (default: 10)')
parser.add_argument('--num_hidden_layers', type=int, default=2,
                    help='num hidden_layers (default: 0)')
parser.add_argument('--flip_rate_fixed', type=float,
                    help='fixed flip rates.', default=0.4)
parser.add_argument('--train_frac', default=1.0, type=float,
                    help='training sample size fraction')
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--trainval_split',  default=0.8, type=float,
                    help='training set ratio')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--dataset', default="balancescale", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')


# added by yz:
parser.add_argument('--pca_k', type=int, default=5,
                    help='PCA dimension (default: 5)')
parser.add_argument('--sample_size', type=int, default=20,
                    help='randomly select samples for analysis')
parser.add_argument('--near_percentage', type=float, default=0.1,
                    help='percentage nearby in terms of L2 norm')

arch_dict = {"FashionMNIST": "resnet18", "cifar10": "resnet18", "cifar100": "resnet34", "mnist": "Lenet",
             "balancescale": "NaiveNet", "krkp": "NaiveNet", "splice": "NaiveNet", "yxguassian": "NaiveNet"}

# load dataset


def main():
    args = parser.parse_args()
    base_dir = "./"+args.dataset+"/"+args.noise_type + \
        str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
    print(args)

    if args.seed is not None:
        fix_seed(args.seed)
    train_val_loader, train_loader, val_loader, est_loader, test_loader = load_ucidata(
        dataset=args.dataset,
        noise_type=args.noise_type,
        random_state=args.seed,
        batch_size=args.batch_size,
        add_noise=True,
        flip_rate_fixed=args.flip_rate_fixed,
        trainval_split=args.trainval_split,
        train_frac=args.train_frac,
        augment=False
    )
    # test_dataset = test_loader.dataset
    # val_dataset = val_loader.dataset

    train_dataset = train_loader.dataset
    #print("training set length is: "+str(len(train_dataset.dataset.data)))
    print("done")

    # ---- Load data ---- #

    noisy_data_list = []
    noisy_rate_list = np.arange(0.1, 1, 0.2)

    # primeY is obtained from K-means unsupervised learning -> we use this as esitmated Clean Y
    primeY = run_kmeans2(train_dataset.dataset)

    # MODIFIED: directly add SYM noise on the noise label

    initial_noise_labels = train_dataset.dataset.targets
    num_classes = train_dataset.dataset._get_num_classes()
    error_list = []

    for noise_rate in noisy_rate_list:
        train_noisy_labels, _, _ = noisify_multiclass_symmetric(initial_noise_labels.copy(
        )[:, np.newaxis], noise_rate, random_state=args.seed, nb_classes=num_classes)

        # calcualte error rate on the noisy labels
        res2 = (primeY == train_noisy_labels[:, 0])
        count2 = collections.Counter(res2)
        error_rate = count2[False]/(count2[False]+count2[True])
        error_list.append(error_rate)

    # for i in range(len(noisy_data_list)):
    #     noisy_data = noisy_data_list[i]
    #     noisy_Y = noisy_data.dataset.targets
    #     res = (primeY == noisy_Y)
    #     count = collections.Counter(res)
    #     error_rate = count[False]/(count[False]+count[True])
    #     error_list.append(error_rate)

    df = pd.concat([pd.DataFrame(noisy_rate_list),
                    pd.DataFrame(error_list)], axis=1)
    df.columns = ['noisy_rate', 'error_rate']

    df['dataset'] = args.dataset

    df['noise_type'] = args.noise_type
    # if dataset in "krkp", "balancescale", "splice", "xyguassian" then it is causal
    df['causal'] = df['dataset'].apply(
        lambda x: 1 if x in ["krkp", "balancescale", "splice", "xyguassian"] else 0)
    df['inital_noise'] = args.flip_rate_fixed
    df['seed'] = args.seed
    df.to_csv('./results/results_method1_error_rate.csv',
              mode='a', index=False, header=False)

    print("all done")


if __name__ == "__main__":
    main()
