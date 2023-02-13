
__all__ = ['get_torchvision_data', 'MNISTDataset', 'get_train_val_dataloaders', 'get_dataloader',
           'AugmentedMNISTDataset', 'get_data']

import os
import sys
import shutil
from pathlib import Path
import random
from collections import namedtuple

import torch
import torchvision
import torch.nn.functional as F
import numpy as np


from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler

import ariamis.utils.all_utils as utils


def get_torchvision_data(dataset_name, directory = "../data") -> Path:
    """
    Returns path to a torchvision dataset. Downloads using torchvision if not found in data_root.

    Args:
        dataset_name: str - currently only MNIST, KMNIST or FashionMNIST
        data_root  : str - Location where data is stored, default is "../data"

    Returns:
        pathlib.Path to dataset, e.g PosixPath('../data/FashionMNIST')

    Note: think some of the torchivison datasets have slightly different formats, for
    example Imagenet has a "split" arg not train. Therefore leaving in this more
    verbose form for now.
    """
    data_root = Path(directory)
    if dataset_name == 'MNIST':
        torchvision.datasets.MNIST(root=data_root, train=True, download=True)
        torchvision.datasets.MNIST(root=data_root, train=False, download=True)
    elif dataset_name == 'KMNIST':
        torchvision.datasets.KMNIST(root=data_root, train=True, download=True)
        torchvision.datasets.KMNIST(root=data_root, train=False, download=True)
    elif dataset_name == 'FashionMNIST':
        torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True)
        torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True)
    else:
        print('Unsupported Dataset string')
        raise
    return  data_root/dataset_name

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path, flatten=False, permute=False, to_device=True):
        """
        Class representing MNIST-like dataset (i.e. also Kuzushiji or Fashion).

        Intended use is to represent full training or test sets.
        Use torch.utils.data.sampler.SubsetRandomSampler
        to obtain a validation set (when contructing dataloaders).

        Args:
            path : path to the data.pt file,
                    e.g. '../data/MNIST/processed/training.pt'
            flatten: flattens data so rows are stacked
            to_device: If true puts tensors on device returned by utils.get_device()
            permute: To implement
        """
        self.loadpath = path
        self.x, self.y = torch.load(path)
        self.x = self.x.float().div(255)
        self.n_classes = len(self.y.unique())

        if flatten:
            self.x = self.x.reshape(self.x.shape[0],-1).contiguous()

        if to_device:
            device = utils.get_device()
            self.x = self.x.to(device)
            self.y = self.y.to(device)

        if permute:
            raise

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def data_shape(self):
        # grabs first datapoint which will be (x,y) tuple so [0] again for x
        return tuple(self[0][0].shape)

    @property
    def n_pixels(self):
        return np.prod(self.data_shape)

    def __repr__(self):
        s = super().__repr__()
        s += '\nData loaded from: '+str(self.loadpath)
        s += '\n'
        s += f'Data is {self.__len__()} examples of shape {self.data_shape}'
        return s


def get_train_val_dataloaders(dataset,
                              batch_size,
                              validation_size=10000,
                              num_workers=0,
                              pin_memory=False,
                              seed=None):
    """
    Splits dataset into a training and validation sets.

    Creates two SubsetRandomSamplers (see https://pytorch.org/docs/1.1.0/data.html)
    Therefore batches will always be shuffled.

    Args:
        - seed: int or None, if not None the data indices used for the validation set are shuffled.
        - pin memory: DataLoader allocate the samples in page-locked memory, which speeds-up the transfer
                      Set true if dataset is on CPU, False if data is already pushed to the GPU.
        - num_workers: if 0 main process does the dataloading.
    """
    if validation_size == 0:
        print("Validation size is 0!")
        raise

    dataset_size  = len(dataset)
    data_indices = list(range(dataset_size))

    if seed:
        np.random.seed(seed)
    # shuffle which images are used as validation
    np.random.shuffle(data_indices) # in-place

    split = validation_size
    train_idx = data_indices[split:]
    valid_idx = data_indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size,
                                               sampler=valid_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    return train_loader, valid_loader

def get_dataloader(dataset,
                   batch_size,
                   shuffle=False,
                   num_workers=0,
                   pin_memory=False):
    """
    A straightforward call to torch.utils.data.DataLoader

    Args:
        - shuffle: bool, whether data order is shuffled (each epoch).
        - pin memory: DataLoader allocate the samples in page-locked memory, which speeds-up the transfer
                      Set true if dataset is on CPU, False if data is already pushed to the GPU.
        - num_workers: if 0 main process does the dataloading.
    """
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size,
                                             sampler=sampler,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory)

    return dataloader


class AugmentedMNISTDataset(MNISTDataset):
    def __init__(self, path, flatten=False, permute=False, to_device=True):
        """
        Class representing MNIST-like dataset with hardcoded augmentation.

        see MNISTDataset docs
        """
        # here we pass flatten=false to MNISTDataset superclass as padding is 2d
        super(AugmentedMNISTDataset, self).__init__(path=path, flatten=False,
                                                    permute=permute, to_device=to_device)

        print('Warning: augmentation not tested fully yet')
        self.x = F.pad(input=self.x, pad=(2, 2, 2, 2), mode='constant', value=0)

        self.flatten = flatten # used in __getitem__


    def __getitem__(self, idx):
        # remember random is faster than numpy.random for non np.arrays.
        xs = random.choice([-2,-1,0,1,2])
        ys = random.choice([-2,-1,0,1,2])
        x1, x2 = 2+xs, 30+xs
        y1, y2 = 2+ys, 30+ys
        x_crop = self.x[idx,x1:x2,y1:y2]
        if self.flatten:
            x_crop = x_crop.reshape(-1)
        return x_crop, self.y[idx]



def get_data(dataset_name, batch_size, directory="../data", seed=7,
             to_device=False, validation_size=10000, flatten=False,
             copy_to_slurmtmpdir=True, test_set=False):
    """
    Convenience function to wrap up steps to get data. Returns two dataloaders for training and eval

    Args:
        test_set - Bool: If True, validation size is not used, full training set loader
                     and test loader are returned
        to_device - Bool: Whether to pre-push the data onto the GPU

    If running locally set copy_to_slurmtmpdir to false (as this won't exist). You might also want
    to set the directory to ./data, not ..
    """
    data_dir = get_torchvision_data(dataset_name, directory)
    if copy_to_slurmtmpdir:
        data_dir = utils.copy_folder_to_slurm_tmpdir(data_dir)

    train_dataset = MNISTDataset(data_dir/"processed"/"training.pt", flatten, to_device=to_device)
    if test_set:
        test_dataset = MNISTDataset(data_dir/"processed"/"test.pt", flatten, to_device=to_device)
        test_loader  = get_dataloader(test_dataset,batch_size,  shuffle=False)
        train_loader = get_dataloader(train_dataset,batch_size, shuffle=True) # Should be true
        return train_loader, test_loader

    else:
        train_loader, valid_loader = get_train_val_dataloaders(train_dataset,
                                                               batch_size,
                                                               validation_size=validation_size,
                                                               seed=seed)
        return train_loader, valid_loader
