import argparse
from random import seed
from mylib.utils import fix_seed
from mylib.data.data_loader import load_ucidata
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
from enum import unique
import random
import math
import collections


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
parser.add_argument('--dataset', default="krkp", type=str,
                    help='db')
parser.add_argument('--select_ratio', default=0, type=float,
                    help='confidence example selection ratio')
parser.add_argument('--pretrained', default=0, type=int,
                    help='using pretrained model or not')

# added by yz:
parser.add_argument('--pca_k', type=int, default=5,
                    help='PCA dimension (default: 5)')
parser.add_argument('--sample_size', type=int, default=20,
                    help = 'randomly select samples for analysis')                  
parser.add_argument('--near_percentage', type=float, default=0.1,
                    help='percentage nearby in terms of L2 norm')

arch_dict = {"FashionMNIST":"resnet18","cifar10":"resnet18","cifar100":"resnet34","mnist":"Lenet","balancescale":"NaiveNet","krkp":"NaiveNet","splice":"NaiveNet","yxguassian":"NaiveNet"}

def runPCA(dataset, n_components):
    X = dataset.data
    pca = PCA(n_components)
    return pca.fit(X)

def entropy(labels):
    freqdist = collections.Counter(labels)
    probs = [freqdist[label] / len(labels) for label in freqdist]
    return -sum(p * math.log(p, 2) for p in probs)

def main():
    args = parser.parse_args()
    base_dir = "./"+args.dataset+"/"+args.noise_type+str(args.flip_rate_fixed)+"/"+str(args.seed)+"/"
    print(args)
    
    if args.seed is not None:
        fix_seed(args.seed)
    train_val_loader, train_loader, val_loader, est_loader, test_loader = load_ucidata(
        dataset = args.dataset,  
        noise_type = args.noise_type,
        random_state = args.seed, 
        batch_size = args.batch_size, 
        add_noise = True, 
        flip_rate_fixed = args.flip_rate_fixed, 
        trainval_split = args.trainval_split,
        train_frac = args.train_frac,
        augment=False,
    )
    test_dataset = test_loader.dataset
    val_dataset = val_loader.dataset

    train_dataset = train_loader.dataset
    k=args.pca_k
    iteration_number = args.sample_size
    near_percentage = args.near_percentage    
    # --- PCA --- #
    pca = runPCA(train_dataset.dataset, k)
    X_pca = pca.transform(train_dataset.dataset.data)

    k = args.pca_k
    iteration_number = args.sample_size
    perc_closest = args.near_percentage

    if len(np.unique(train_dataset.dataset.targets))==2:
        is_binary = True
    else:
        is_binary = False

    random_index_list = []
    entropy_list = []
    ratio_list =[]
    for i in range(iteration_number):

        
        while (1):
            random_index = random.randint(0, len(train_dataset.dataset.data))
            if (random_index not in random_index_list):
                random_index_list.append(random_index)
                break
        
        l2_norm = np.linalg.norm(X_pca - X_pca[random_index], axis=1)
        # find the index of examples that are close to the randomly selected example
        close_samples_index = np.argsort(l2_norm)[:int(len(train_dataset.dataset.data)*perc_closest)]
        # calculate the entropy among these samples
        close_entropy = entropy(train_dataset.dataset.targets[close_samples_index])
        entropy_list.append(close_entropy)

        # for binary labels, calculate the ratio of count the unique values of targets
        if (is_binary == True):
            unique, counts = np.unique(train_dataset.dataset.targets[close_samples_index], return_counts=True)
            ratio = counts[0]/(counts[0]+counts[1])
            ratio_list.append(ratio)
        else:
            ratio_list.append(0)
        
    # concatenate the random index, entropy and ratio lists into a dataframe
    df = pd.DataFrame(list(zip(random_index_list, entropy_list, ratio_list)), columns =['random_index', 'entropy', 'class0_ratio'])

    y_axis_label_left = "Entropy"
    y_axis_label_right = "Ratio of class 0"
    title = "PCA (k="+str(k)+") on "+args.dataset+" dataset"+" with "+str(args.flip_rate_fixed)+" "+str(args.noise_type)+" noise"+'\n' + "Randomly selected "+str(iteration_number)+" samples within "+str(perc_closest*100)+"% of its neighbors (L2 norm)"

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(df.index,
            df.entropy,
            color="red", 
            marker="o")
    # set x-axis label
    ax.set_xlabel("random samples", fontsize = 14)
    # set y-axis label
    ax.set_ylabel(y_axis_label_left,
                color="red",
                fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(df.index, df.class0_ratio,color="blue",marker="o")
    ax2.set_ylabel(y_axis_label_right,color="blue",fontsize=14)

    #ax.set_ylim(0,1.1)
    #ax2.set_ylim(0,1.1)
    plt.title (title)
    plt.show()

    # save the plot as a file

    output_path = "./results/"+str(args.dataset)+"_PCA_"+str(args.pca_k)+"_random_"+str(args.sample_size)+"_near_"+str(perc_closest)+"_"+str(args.noise_type)+"_"+str(args.flip_rate_fixed)+"_noise"+".png"
    fig.savefig(output_path,
                format='jpeg',
                dpi=100,
                bbox_inches='tight')    
    print("Done")

if __name__ == "__main__":
    main()
