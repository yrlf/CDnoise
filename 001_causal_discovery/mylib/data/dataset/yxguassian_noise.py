

import time
import warnings
from scipy.stats import multivariate_normal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.utils.data as Data
from PIL import Image
import os
# from mylib.noise.generator import CCN_generator
from .util import noisify
import torch

__all__ = ["yxguassian_noise"]

def get_Y_star(x, clean_Y ):
	mn_neg = multivariate_normal(mean=[0]*5, cov=np.eye(5))
	mn_pos = multivariate_normal(mean=[1]*5, cov=np.eye(5))
	neg_density = mn_neg.pdf(x)
	pos_density = mn_pos.pdf(x)
	x_density = neg_density+pos_density
	neg_post = neg_density/x_density
	pos_post = pos_density/x_density
	dist = np.array([neg_post,pos_post])
	dist = dist.T

	pred = torch.max(torch.from_numpy(dist), 1)[1]
	eval_correct = (pred == torch.from_numpy(clean_Y)).sum()

	##
	return pred.numpy().astype(np.long)

class yxguassian_noise(Data.Dataset):


    def __init__(self,
        root="./dataset/krkp", 
        train = True,
        transform=None, 
        transform_eval = None,
        target_transform=None, 
        add_noise= True, 
        flip_rate_fixed = None, 
        noise_type = '', 
        random_state = 1):
        


        self.transform = transform
        self.transform_eval = transform_eval
        self.target_transform = target_transform
        self.t_matrix = None
        root = os.path.expanduser(root)

        if train == True:
            self.root = root+"_train.csv"
        else:
            self.root = root+"_test.csv"    

        self.apply_transform_eval = False

        self.data_info = pd.read_csv(self.root,header=None)
        label_index = self.data_info.columns[-1]
        # self.data = np.asarray(self.data_info.iloc[:, self.data_info.columns != label_index]).astype(np.float32)
        # self.targets = np.asarray(self.data_info.iloc[:, label_index])
        tmp_arr = np.asarray(self.data_info.values.tolist())
        # self.data = np.asarray(self.data_info[:,:-1].astype(np.float32)).astype(np.float32)
        # self.targets = np.asarray(self.data_info[:,-1].astype(np.long))     
       
        self.data = tmp_arr[:,:-1].astype(np.float32)
        self.targets = tmp_arr[:,-1].astype(np.long)


        self.clean_targets = self.targets.copy()
        self.hat_clean_targets  = self.clean_targets.copy()
        # get_Y_star(self.data, self.clean_targets  )
  
        # self.data = self.data.astype(np.float32)
   
        if add_noise:
            noisy_targets, self.actual_noise_rate, self.t_matrix = noisify(
                dataset=zip(torch.from_numpy(self.data).float(), torch.from_numpy(self.targets)), 
                train_labels=self.targets, 
                noise_type=noise_type, 
                noise_rate=flip_rate_fixed, 
                random_state=random_state,
                nb_classes=self._get_num_classes(),
                feature_size=len(self.data[0])
            )
            noisy_targets = noisy_targets.squeeze()
            self._set_targets(noisy_targets)
  
        self.is_confident = np.zeros(len(self.clean_targets))
        # self.hat_clean_targets = self.targets.copy()
       

    def __getitem__(self, index):
        img, target, clean_target, hat_clean_target, confidenice = self.data[index], int(self.targets[index]), int(self.clean_targets[index]), int(self.hat_clean_targets[index]), int(self.is_confident[index])
        # if self.apply_transform_eval:
        #     transform = self.transform_eval
        # else:
        #     transform = self.transform  

        # if self.transform is not None:
        #     img = transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #     clean_target = self.target_transform(clean_target)
        #     hat_clean_target = self.target_transform(hat_clean_target)
        #     confidenice = self.target_transform(confidenice)

        return img, target, clean_target, hat_clean_target, confidenice
     
    def _set_targets(self,n_targets):
        self.targets = n_targets

    def _get_num_classes(self):
        return len(set(self.targets))

    def _get_targets(self):
        return self.targets

    def eval(self):
        self.apply_transform_eval = True

    def train(self):
        self.apply_transform_eval = False

    def __len__(self):
        return len(self.targets)

    def get_clean_ratio(self):
        correct = 0
        t_number = 0
        for (c_label, h_c_label, confidence) in zip(self.clean_targets, self.hat_clean_targets,  self.is_confident):
            if confidence == 1:
                if c_label == h_c_label:
                    correct +=1
                t_number +=1
        return correct/(t_number+1e-10)
