import argparse
from mylib.utils import fix_seed
from mylib.data.data_loader import load_ucidata
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import PCA
import collections
import random
import csv
import math

from kmeans import run_kmeans
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
                    help='randomly select samples for analysis')
parser.add_argument('--near_percentage', type=float, default=0.1,
                    help='percentage nearby in terms of L2 norm')

arch_dict = {"FashionMNIST": "resnet18", "cifar10": "resnet18", "cifar100": "resnet34", "mnist": "Lenet",
             "balancescale": "NaiveNet", "krkp": "NaiveNet", "splice": "NaiveNet", "yxguassian": "NaiveNet"}


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
        augment=False,
    )
    test_dataset = test_loader.dataset
    val_dataset = val_loader.dataset

    train_dataset = train_loader.dataset
    k = args.pca_k
    iteration_number = args.sample_size
    near_percentage = args.near_percentage
    # --- PCA --- #
    # normalizaion of train_dataset using z score sci-ki learn

    train_dataset.dataset.data = (train_dataset.dataset.data - np.mean(
        train_dataset.dataset.data, axis=0)) / np.std(train_dataset.dataset.data, axis=0)

    pca = runPCA(train_dataset.dataset, k)
    X_pca = pca.transform(train_dataset.dataset.data)

    k = args.pca_k
    iteration_number = args.sample_size
    perc_closest = args.near_percentage

    if len(np.unique(train_dataset.dataset.targets)) == 2:
        is_binary = True
    else:
        is_binary = False

    random_index_list = []
    entropy_list = []
    ratio_list = []
    for i in range(iteration_number):

        while (1):
            random_index = random.randint(0, len(train_dataset.dataset.data))
            if (random_index not in random_index_list):
                random_index_list.append(random_index)
                break

        l2_norm = np.linalg.norm(X_pca - X_pca[random_index], axis=1)
        # find the index of examples that are close to the randomly selected example
        close_samples_index = np.argsort(l2_norm)[:int(
            len(train_dataset.dataset.data)*perc_closest)]
        # calculate the entropy among these samples
        close_entropy = entropy(
            train_dataset.dataset.targets[close_samples_index])
        entropy_list.append(close_entropy)

        # for binary labels, calculate the ratio of count the unique values of targets
        if (is_binary == True):
            unique, counts = np.unique(
                train_dataset.dataset.targets[close_samples_index], return_counts=True)
            ratio = counts[0]/(counts[0]+counts[1])
            ratio_list.append(ratio)
        else:
            ratio_list.append(0)

    # concatenate the random index, entropy and ratio lists into a dataframe
    df = pd.DataFrame(list(zip(random_index_list, entropy_list, ratio_list)), columns=[
                      'random_index', 'entropy', 'class0_ratio'])

    if args.dataset in ["krkp", "balancescale", "splice"]:
        is_causal = 1
    else:
        is_causal = 0

    output_filename = "./results/pca_results.csv"
    with open(output_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.dataset, args.pca_k, args.sample_size, perc_closest, args.noise_type, args.flip_rate_fixed,
                        df['entropy'].mean(), df['entropy'].var(), df['class0_ratio'].mean(), df['class0_ratio'].var(), is_causal] )

    print("Done")


if __name__ == "__main__":
    main()
