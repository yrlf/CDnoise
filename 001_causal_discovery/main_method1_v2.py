from errno import errorcode
import sys
sys.path.insert(0, './')
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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections
import numpy as np
from fixmatch.utils import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from PIL import Image
import torchvision
from torchvision import datasets, transforms
# adjust the load data function to load training data only
from mylib.data.data_loader.load_ucidata import load_ucidata2
from cgi import test
from torch.utils.data.sampler import SubsetRandomSampler
from mylib.data.data_loader.load_cifardata import load_cifardata
from mylib.data.data_loader.load_mnistdata import load_mnistdata
import mylib.data.dataset
from mylib.data.dataset.util import noisify_multiclass_symmetric

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from cal_acc import *
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

# from SPICE.fixmatch.datasets.data_utils import get_data_loader
# from SPICE.fixmatch.datasets.ssl_dataset_robust import SSL_Dataset
# from SPICE.fixmatch.utils import net_builder

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot
        
        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
                
    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
            
        #set augmented images
            
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target, idx
            else:
                return img_w, self.strong_transform(img), target

    
    def __len__(self):
        return len(self.data)

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
    #idx2 = _hungarian_match(identified_clusters, clean_Y)
    prime_Y = get_prime_Y(tilde_Y, identified_clusters, idx2[1])
    # yz: directly return prime_Y without using count_m
    return prime_Y


mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [0.485, 0.456, 0.406]
mean['npy'] = [0.485, 0.456, 0.406]
mean['npy224'] = [0.485, 0.456, 0.406]
mean['FashionMNIST'] = [0.2860]


std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [0.229, 0.224, 0.225]
std['npy'] = [0.229, 0.224, 0.225]
std['npy224'] = [0.229, 0.224, 0.225]
std['FashionMNIST'] = [0.3530]

def get_dset(data,targets, num_classes, name, train, use_strong_transform=False, 
                strong_transform=None, onehot=False):
    """
    get_dset returns class BasicDataset, containing the returns of get_data.
    
    Args
        use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
        strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
        onehot: If True, the label is not integer, but one-hot vector.
    """
    
    transform = get_transform(mean[name], std[name], name, train)
    
    return BasicDataset(data, targets, num_classes, transform, 
                        use_strong_transform, strong_transform, onehot)
    

    
def get_transform(mean, std, dataset, train=True):
    if dataset in ['cifar10', 'cifar20', 'cifar100']:
        crop_size = 32
    elif dataset in ['stl10', 'npy']:
        crop_size = 96
    elif dataset in ['FashionMNIST']:
        crop_size = 28
    elif dataset in ['npy224']:
        crop_size = 224
    else:
        raise TypeError
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(crop_size, padding=4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean, std)])
        
def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 4,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory)
    else:
        print("data sample is not implmented")

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def cross_entropy_error(y,t):
    delta=1e-7 
    import torch.nn as nn
    from torch.autograd import Variable
    #y=y.flatten()
    # convert y to one hot encoding
    y=one_hot(y,2)
    y = Variable(torch.FloatTensor(y))
    
    #t = np.expand_dims(t, axis=1)

    t = Variable(torch.LongTensor(t))
    t = Variable(torch.FloatTensor(y.shape[0]).uniform_(0, 1).long())
    print(t)
    print("============")
    print(y)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y, t)
    print(loss)
    return loss.item()
    
    y=y.flatten()
    print(y.shape[0])
    print(t.shape[0])
    return -np.sum(y*np.log(t+delta))/y.shape[0]


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


'''
Backbone Net Configurations
'''
parser.add_argument('--net', type=str, default='WideResNet')
parser.add_argument('--net_from_name', type=bool, default=False)
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

'''
Data Configurations
'''
#parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--data_dir', type=str, default='./datasets/cifar-10-batches-py')
#parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--label_file', type=str, default=None)
parser.add_argument('--all', type=int, default=0)
parser.add_argument('--unlabeled', type=bool, default=False)

parser.add_argument('--output', type =str, default="temp")

arch_dict = {"FashionMNIST": "resnet18", "cifar10": "resnet18", "cifar100": "resnet34", "mnist": "Lenet",
             "balancescale": "NaiveNet", "krkp": "NaiveNet", "splice": "NaiveNet", "yxguassian": "NaiveNet"}

# load dataset

args = parser.parse_args()
if args.dataset == "cifar10" or args.dataset == "cifar10n_worst" or args.dataset == "cifar10n_aggre" or args.dataset == "cifar10n_random1" or args.dataset =='cifar10n_random2' or args.dataset == "cifar10n_random3":
    if args.seed is not None:
        fix_seed(args.seed)
        train_val_loader, train_loader, val_loader, est_loader, test_loader = load_cifardata(
            dataset=args.dataset, 
            random_state=args.seed,
            add_noise=True,
            batch_size=args.batch_size, 
            train_frac=args.train_frac, 
            trainval_split=args.trainval_split, 
            noise_type=args.noise_type, 
            flip_rate_fixed=args.flip_rate_fixed, 
            augment=False
        )
elif args.dataset == "imagenet10":
    pass

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
    primeY =[]
    if args.dataset == "cifar10" or args.dataset =="cifar10n_worst" or args.dataset =="cifar10n_aggre" or args.dataset =="cifar10n_random1" or args.dataset == "cifar10n_random2" or args.dataset =="cifar10n_random3":
        if (args.dataset == "cifar10"):
            #  convert the data to suitable format (feeding into pretrained SPICE* model)
            my_data_dest = get_dset(train_dataset.dataset.data,train_dataset.dataset.targets,10,"cifar10",True)
            my_data_loader=get_data_loader(my_data_dest,batch_size=args.batch_size,num_workers=0)
            #  load the pretrained SPICE* model

        elif (args.dataset == "cifar10n_worst"):
            # load real world noise labels
            noise_file = torch.load('./cifar-10-100n/data/CIFAR-10_human.pt')
            clean_label = noise_file['clean_label']
            worst_label = noise_file['worse_label']
            my_data_dest = get_dset(train_dataset.dataset.data, worst_label, 10, "cifar10", True)
            my_data_loader = get_data_loader(my_data_dest, batch_size=args.batch_size, num_workers=0)
        elif (args.dataset == "cifar10n_aggre"):
            noise_file = torch.load('./cifar-10-100n/data/CIFAR-10_human.pt')
            clean_label = noise_file['clean_label']
            aggre_label = noise_file['aggre_label']
            my_data_dest = get_dset(train_dataset.dataset.data, aggre_label, 10, "cifar10", True)
            my_data_loader = get_data_loader(my_data_dest, batch_size=args.batch_size, num_workers=0)
        elif (args.dataset == "cifar10n_random1"):
            noise_file = torch.load('./cifar-10-100n/data/CIFAR-10_human.pt')
            clean_label = noise_file['clean_label']
            random1_label = noise_file['random_label1']
            my_data_dest = get_dset(train_dataset.dataset.data, random1_label, 10, "cifar10", True)
            my_data_loader = get_data_loader(my_data_dest, batch_size=args.batch_size, num_workers=0)
        elif (args.dataset == "cifar10n_random2"):
            noise_file = torch.load('./cifar-10-100n/data/CIFAR-10_human.pt')
            clean_label = noise_file['clean_label']
            random2_label = noise_file['random_label2']
            my_data_dest = get_dset(train_dataset.dataset.data, random2_label, 10, "cifar10", True)
            my_data_loader = get_data_loader(my_data_dest, batch_size=args.batch_size, num_workers=0)
        elif (args.dataset == "cifar10n_random3"):
            noise_file = torch.load('./cifar-10-100n/data/CIFAR-10_human.pt')
            clean_label = noise_file['clean_label']
            random3_label = noise_file['random_label3']
            my_data_dest = get_dset(train_dataset.dataset.data, random3_label, 10, "cifar10", True)
            my_data_loader = get_data_loader(my_data_dest, batch_size=args.batch_size, num_workers=0)

        model = torch.load('model_cifar10_cls.pth')
        load_model = model['train_model']
        for k in list(load_model.keys()):
            # Initialize the feature module with encoder_q of moco.
            if k.startswith('model.'):
                # remove prefix
                load_model[k[len('model.'):]] = load_model[k]

                del load_model[k]
                # print(k)
        # buil network
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'dropRate': args.dropout})

        net = _net_builder(num_classes=10)
        net.load_state_dict(load_model,strict=False)
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        labels_pred = []
        labels_gt = []

        scores = []
        with torch.no_grad():
            for image, target,_ in my_data_loader:
                image = image.type(torch.FloatTensor).cuda()
                logit = net(image)
                scores.append(logit.cpu().numpy())
                labels_pred.append(torch.max(logit, dim=-1)[1].cpu().numpy())
                labels_gt.append(target.cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels_pred = np.concatenate(labels_pred, axis=0)
        labels_gt = np.concatenate(labels_gt, axis=0)


        try:
            acc = calculate_acc(labels_pred, labels_gt)
        except:
            acc = -1

        print(f"Test Accuracy: {acc}")     
        # get prime_Y from pretrained SPICE* model
        if (args.dataset == "cifar10"):
            #idx2 = _hungarian_match(labels_pred, train_dataset.dataset.targets)
            idx2 = _hungarian_match(labels_pred, train_dataset.dataset.clean_targets)
            primeY = get_prime_Y(train_dataset.dataset.targets, labels_pred, idx2[1])
        elif (args.dataset == "cifar10n_worst"):
            #idx2 = _hungarian_match(labels_pred, worst_label)
            idx2 = _hungarian_match(labels_pred, clean_label)
            primeY = get_prime_Y(worst_label, labels_pred, idx2[1])
        elif (args.dataset == "cifar10n_aggre"):
            #idx2 = _hungarian_match(labels_pred, aggre_label)
            idx2 = _hungarian_match(labels_pred, clean_label)
            primeY = get_prime_Y(aggre_label, labels_pred, idx2[1])
        elif (args.dataset == "cifar10n_random1"):
            #idx2 = _hungarian_match(labels_pred, random1_label)
            idx2 = _hungarian_match(labels_pred, clean_label)
            primeY = get_prime_Y(random1_label, labels_pred, idx2[1])
        elif (args.dataset == "cifar10n_random2"):
            #idx2 = _hungarian_match(labels_pred, random2_label)
            idx2 = _hungarian_match(labels_pred, clean_label)
            primeY = get_prime_Y(random2_label, labels_pred, idx2[1])
        elif (args.dataset == "cifar10n_random3"):
            #idx2 = _hungarian_match(labels_pred, random3_label)
            idx2 = _hungarian_match(labels_pred, clean_label)
            primeY = get_prime_Y(random3_label, labels_pred, idx2[1])
        
        #idx2 = _hungarian_match(labels_pred, train_dataset.dataset.clean_targets)
        #primeY = get_prime_Y(train_dataset.dataset.clean_targets, labels_pred, idx2[1])
    

        

    else:
        # primeY is obtained from K-means unsupervised learning -> we use this as esitmated Clean Y
        primeY = run_kmeans2(train_dataset.dataset)

    # MODIFIED: directly add SYM noise on the noise label
    if (args.dataset == "cifar10"):
        initial_noise_labels = train_dataset.dataset.targets
    elif (args.dataset == "cifar10n_worst"):
        initial_noise_labels = worst_label
    elif (args.dataset == "cifar10n_aggre"):
        initial_noise_labels = aggre_label
    elif (args.dataset == "cifar10n_random1"):
        initial_noise_labels = random1_label
    elif (args.dataset == "cifar10n_random2"):
        initial_noise_labels = random2_label
    elif (args.dataset == "cifar10n_random3"):
        initial_noise_labels = random3_label
    else:
        initial_noise_labels = train_dataset.dataset.targets
    num_classes = train_dataset.dataset._get_num_classes()
    error_list = []

    noisy_data_list = []
    noisy_rate_list = np.arange(0.1, 1, 0.1)

    for noise_rate in noisy_rate_list:
        train_noisy_labels, _, _ = noisify_multiclass_symmetric(initial_noise_labels.copy(
        )[:, np.newaxis], noise_rate, random_state=args.seed+1, nb_classes=num_classes)

        if args.dataset == "DEBUG-yxguassian":
            clean_labels = train_dataset.dataset.clean_targets
            # delist the train_noisy_labels
            train_noisy_labels2 = [item for sublist in train_noisy_labels for item in sublist]
            
            #print(f"train noise label legnth is: ",len(train_noisy_labels))
            #print(f"clean label legnth is: ",len(clean_labels))
            #acc1 = (train_noisy_labels2==clean_labels).sum()/len(clean_labels)
            #acc2 = (initial_noise_labels==clean_labels).sum()/len(clean_labels)
            
            #print(train_noisy_labels)

            #print(f"total number of same labels between train_noisy_labels and clean labels", (train_noisy_labels==clean_labels).sum())

            #print(f"total number of same labels between inital_noise_labels and clean labels", (initial_noise_labels==clean_labels).sum())

            #print("noise rate is: ", noise_rate)
            #print(f"train_noisy_labels vs. clean Y: {acc1}")
            #print(f"initial_noise_labels vs. clean Y: {acc2}")

        

        # calcualte error rate on the noisy labels
        res2 = (primeY == train_noisy_labels[:, 0])
        count2 = collections.Counter(res2)
        error_rate = count2[False]/(count2[False]+count2[True])
        #print("error rate is", error_rate)

        # #error_rate = cross_entropy_error(train_noisy_labels, primeY)
        # # count number of 0 in primeY
        # pcount = collections.Counter(primeY)
        # ncount = collections.Counter(train_noisy_labels[:, 0])
        
        # py0 = pcount[0]/(pcount[0]+pcount[1])
        # py1 = pcount[1]/(pcount[0]+pcount[1])

        # ny0= ncount[0]/(ncount[0]+ncount[1])
        # ny1= ncount[1]/(ncount[0]+ncount[1])

        # error_rate = - ( py0*np.log(ny0) + py1*np.log(ny1) )  

        error_list.append(error_rate)

        # calculate cross-entropy between primeY and train_noisy_labels

    

    #exit()
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
    

    
    filename ='./results/results_'+args.output+'.csv'
    # check whether file exists
    if os.path.exists(filename) ==False:
        df.to_csv(filename, header=True, index=False)
    else:
        df.to_csv(filename,
              mode='a', index=False, header=False)

    print("all done")


if __name__ == "__main__":
    main()
