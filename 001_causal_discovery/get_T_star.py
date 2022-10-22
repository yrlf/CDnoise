import numpy as np
from scipy.stats import multivariate_normal
import torch


def get_T_star(dataset):
	data = dataset.data
	noisy_Y = dataset.targets
	Y_star = dataset.hat_clean_targets
	return count_m(noisy_Y,Y_star)
	



def get_Y_star(x,mn_neg,mn_pos,noisy_Y, clean_Y ):
	neg_density = mn_neg.pdf(x)
	pos_density = mn_pos.pdf(x)
	x_density = neg_density+pos_density
	neg_post = neg_density/x_density
	pos_post = pos_density/x_density
	dist = np.array([neg_post,pos_post])
	dist = dist.T

	pred = torch.max(torch.from_numpy(dist), 1)[1]
	eval_correct = (pred == torch.from_numpy(clean_Y)).sum()
	print(eval_correct.item()/len(clean_Y))
	

	##
	return pred.numpy()


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