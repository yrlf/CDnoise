import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, DBSCAN
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

    row, col  = linear_sum_assignment(num_samples - num_correct)
    return row, col 



def run_mkmeans(dataset):
    X = dataset.data.reshape(len(dataset.data), -1)
    print(X)
    clean_Y = dataset.clean_targets
    tilde_Y=dataset.targets
    values, counts = np.unique(clean_Y, return_counts=True)
    kmeans = MiniBatchKMeans(n_clusters=len(values),batch_size=512)
    kmeans.fit(X, tilde_Y)
    identified_clusters = kmeans.fit_predict(X)
    # note that to better match the cluster Id to tilde_Y, 
    # we could use hat_clean Y which obtained by current noise-robust method, but for the simple dataset , it may not necessary  
    idx2 = _hungarian_match(identified_clusters,tilde_Y)
    prime_Y = get_prime_Y(tilde_Y, identified_clusters,idx2[1])
    return count_m(tilde_Y, prime_Y)


def get_prime_Y(noisy_classes, pred_classes, mappings):
    prime_Y = np.zeros(len(noisy_classes))
    for i in range(len(pred_classes)):
        prime_Y[i] = mappings[pred_classes[i]]

    return prime_Y




def count_m(noisy_Y, prime_Y):
    values, counts = np.unique(prime_Y, return_counts=True)
    length = len(values)
    m = np.zeros((length,length))

    for i in range(noisy_Y.shape[0]):
        m[int(prime_Y[i])][int(noisy_Y[i])]+=1

    sum_matrix = np.tile(counts,(len(values),1)).transpose()
    return m/sum_matrix