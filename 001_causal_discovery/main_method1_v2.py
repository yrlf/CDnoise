import scipy
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
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
import sys
sys.path.insert(0, './')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections
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

from SPICE.fixmatch.datasets.data_utils import get_data_loader
from SPICE.fixmatch.datasets.ssl_dataset_robust import SSL_Dataset
from SPICE.fixmatch.utils import net_builder

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

def calculate_acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)


def calculate_nmi(predict_labels, true_labels):
    # NMI
    nmi = metrics.normalized_mutual_info_score(
        true_labels, predict_labels, average_method='geometric')
    return nmi


def calculate_ari(predict_labels, true_labels):
    # ARI
    ari = metrics.adjusted_rand_score(true_labels, predict_labels)
    return ari

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

args = parser.parse_args()
if args.dataset == "cifar10":
    print("load cifar10 model SPICE*")
    load_model = torch.load("/SPICE/cifar10_model.pth")
    for k in list(load_model.keys()):

        # Initialize the feature module with encoder_q of moco.
        if k.startswith('model.'):
            # remove prefix
            load_model[k[len('model.'):]] = load_model[k]

            del load_model[k]
            # print(k)

    if args.net in ['WideResNet', 'WideResNet_stl10', 'WideResNet_tiny', 'resnet18', 'resnet18_cifar', 'resnet34']:
        _net_builder = net_builder(args.net,
                                args.net_from_name,
                                {'depth': args.depth,
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'dropRate': args.dropout})
        
    elif args.net == 'ClusterResNet':
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'input_size': args.input_size})
    else:
        raise TypeError
    
    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    _eval_dset = SSL_Dataset(name=args.dataset, train=False,
                             data_dir=args.data_dir, label_file=None, all=args.all, unlabeled=False)
    # print(args.all)

    eval_dset = _eval_dset.get_dset()
    print(len(eval_dset))

    eval_loader = get_data_loader(eval_dset,
                                  args.batch_size,
                                  num_workers=1)

    acc = 0.0
    labels_pred = []
    labels_gt = []
    scores = []
    with torch.no_grad():
        for image, target, _ in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)

            scores.append(logit.cpu().numpy())

            labels_pred.append(torch.max(logit, dim=-1)[1].cpu().numpy())
            labels_gt.append(target.cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    labels_pred = np.concatenate(labels_pred, axis=0)
    labels_gt = np.concatenate(labels_gt, axis=0)
    # save labels added by yz
    np.save("labels_pred.npy", labels_pred)

    try:
        acc = calculate_acc(labels_pred, labels_gt)
    except:
        acc = -1

    nmi = calculate_nmi(labels_pred, labels_gt)
    ari = calculate_ari(labels_pred, labels_gt)

    print(f"Test Accuracy: {acc}, NMI: {nmi}, ARI: {ari}")

    if args.scores_path is not None:
        np.save(args.scores_path, scores)

else:
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
base_dir = "./"+args.dataset+"/"+args.noise_type + \
    str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
print(args)


def main():

    # ---- Load data ---- #

    noisy_data_list = []
    noisy_rate_list = np.arange(0.1, 1, 0.1)

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
    df.to_csv('./results/results_method1_v2_error_rate_add_noise_model.csv',
              mode='a', index=False, header=False)

    print("all done")


if __name__ == "__main__":
    main()
