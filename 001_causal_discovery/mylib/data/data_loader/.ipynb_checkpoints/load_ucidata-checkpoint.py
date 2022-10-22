import torch
import torchvision
import torchvision.transforms as transforms
from .subset import Subset
import mylib
from mylib.data.data_loader.utils import create_train_val
from .dataloader import DataLoader_noise
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        t = tensor + np.random.normal(self.mean , self.std, size=tensor.size)
        t= t.astype(np.float32)
   
        return t
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


__all__ = ["load_ucidata"]

data_info_dict = {
    "balancescale":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/balancescale/balancescale",
        'random_crop':None
    },
    "splice":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/splice/splice",
        'random_crop':None
    },
    "krkp":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/krkp/krkp",
        'random_crop':None
    },
    "letter":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/letter/letter",
        'random_crop':None
    },
    "waveform":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/waveform/waveform",
        'random_crop':None
    },
    "yxguassian":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/yxguassian/yxguassian",
        'random_crop':None
    },
    "xyguassian":{
        "mean":(0),
        "std":(1),
        "root": "./datasets/xyguassian/xyguassian",
        'random_crop':None
    },
}
        


def load_ucidata(dataset = "CIFAR10", num_workers = 0, batch_size = 128, add_noise = False, noise_type = None, flip_rate_fixed = None, random_state = 1, trainval_split=None, train_frac = 1, augment = False):
  
  

    # print('=> preparing data..')
 
    # dataset = dataset.upper()
    info = data_info_dict[dataset]

    root = info["root"]
    
    transform_strong = transforms.Compose([ AddGaussianNoise(0.,0.1)]) 

    test_dataset = mylib.data.dataset.__dict__["UCL_noise"](root=root, train=False, transform=None, transform_eval=None, transform_strong = transform_strong, add_noise = False, target_transform = None)
    train_val_dataset = mylib.data.dataset.__dict__["UCL_noise"](
        root = root, 
        train = True, 
        transform = None, 
        transform_eval = None,
        transform_strong = transform_strong,
        target_transform = None,
        add_noise = True,
        noise_type = noise_type, 
        flip_rate_fixed = flip_rate_fixed,
        random_state = random_state
    )



    train_dataset, val_dataset = create_train_val(train_val_dataset,trainval_split,train_frac)
    train_val_dataset = Subset(train_val_dataset, list(range(0, len(train_val_dataset), 1))) 
    train_val_loader = DataLoader_noise(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader_noise(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    est_loader = DataLoader_noise(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader_noise(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    

    return train_val_loader, train_loader, val_loader, est_loader, test_loader



