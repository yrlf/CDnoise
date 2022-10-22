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

    row, col  = linear_sum_assignment(num_samples - num_correct)
    return row, col 



def run_kmeans(dataset):


    X = dataset.data
    clean_Y = dataset.clean_targets
    tilde_Y=dataset.targets
    values, counts = np.unique(clean_Y, return_counts=True)

    kmeans = KMeans(n_clusters=len(values))
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
    return prime_Y

# def get_class(true_classes, pred_classes):
#     # true_classes=np.random.randint(2, size=100)
#     # pred_classes=np.random.randint(2, size=100)
#     values, counts = np.unique(pred_classes, return_counts=True)
    
#     length = len(values)
#     m = np.zeros((length,length))
#     m2 = np.array(range(length))
#     y_hat = np.zeros(len(true_classes))


#     for i in range(true_classes.shape[0]):
#         m[true_classes[i]][pred_classes[i]]+=1


#     for i in range(len(m)):
#         temp = -1
#         temp_j = 0
#         temp_i = 0
#         for j in range(len(m)):

#             consistency = m[i][j]/counts[j]
#             # print(m[i][j])
#             # print(counts[j])
       
#             # print(consistency)
#             # print(m[i][j])
#             # print(counts[j])
#             if temp < consistency:
#                 temp = consistency
#                 temp_j = j
#                 temp_i = i
        
        
#         # m2[temp_i] = temp_j


#         m2[temp_j] = temp_i
#                 # print("_____")
#                 # print(j,i)
#         # exit()
#     values, counts = np.unique(m2, return_counts=True)
#     # print(m2.tolist())
#     # print(len(values))

#     for i in range(len(pred_classes)):
#         # print(true_classes[i])
#         # print(m2[true_classes[i]])
#         # print(" ")
#         y_hat[i] = m2[pred_classes[i]]

#     # print(m)
#     # print(m2)
#     # print(y_hat)
#     # return y_hat
#     # print(m)
#     # print(m2)




def count_m(noisy_Y, prime_Y):
    values, counts = np.unique(prime_Y, return_counts=True)
    #print(values)
    length = len(values)
    m = np.zeros((length,length))
    #print(counts)

    for i in range(noisy_Y.shape[0]):
        m[int(prime_Y[i])][int(noisy_Y[i])]+=1

    sum_matrix = np.tile(counts,(len(values),1)).transpose()
    #print(sum_matrix)
    #print(m/sum_matrix)
    return m/sum_matrix