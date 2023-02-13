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

=
def get_torchvision_data(dataset_name, directory = f"{os.environ['HOME']}/data") -> Path:
    """
    This is an extension of the function from data.mnist.

    Returns path to a torchvision dataset. Downloads using torchvision if not found in data_root.

    Args:
        dataset_name: str - currently only MNIST, KMNIST or FashionMNIST
        directory  : str - Location where data is stored, default is "$HOME/data"

    Returns:
        pathlib.Path to dataset basefolder, e.g PosixPath('$HOME/data/FashionMNIST')


    Note: I think some of the torchivison datasets have slightly different formats, for
    example Imagenet has a "split" arg not train. Thereofore making
    """
    data_root = Path(directory)
    if dataset_name == 'MNIST':
        torchvision.datasets.MNIST(root=data_root, train=True, download=True)
        torchvision.datasets.MNIST(root=data_root, train=False, download=True)
        base_folder = 'MNIST'
    elif dataset_name == 'KMNIST':
        torchvision.datasets.KMNIST(root=data_root, train=True, download=True)
        torchvision.datasets.KMNIST(root=data_root, train=False, download=True)
        base_folder = 'KMNIST'
    elif dataset_name == 'FashionMNIST':
        torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True)
        torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True)
        base_folder = 'FashionMNIST'
    elif dataset_name == 'Cifar10':
        d = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
        base_folder = d.base_folder
    elif dataset_name == 'Cifar100':
        d = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True)
        torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
        base_folder = d.base_folder
    elif dataset_name == 'VOC':
        pass
        # not sure about how this gets downloaded
        d = torchvision.datasets.VOCDetection(root=data_root,year='2012',imageset='train', download=True)
        torchvision.datasets.VOCDetection(root=data_root,year='2012',imageset='val', download=True)
    elif dataset_name == 'SVHN':
        pass
        #torchvision.datasets.SVHN()
    else:
        print('Unsupported Dataset string')
        raise
    return  data_root/ base_folder

# Cell

def get_cifar10_train_val_dataloaders(batch_size, val_batch_size, validation_size=10000,
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

        Assumes default
    """
    # first just check it is downloaded
    data_dir = get_torchvision_data(dataset_name='Cifar10') # assume default dir here
    # then copy to slurm
    data_slurm_dir = utils.copy_folder_to_slurm_tmpdir(data_dir)
    #print(data_slurm_dir, data_slurm_dir.parent)

    # load training dataset (as we are doing a split)
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset =  torchvision.datasets.CIFAR10(root=data_slurm_dir.parent, train=True,
                                            transform=dataset_transform, download=True)

    # create train and valid dataloaders from train dataset
    dataset_size  = len(dataset)
    #print(dataset_size)
    data_indices = list(range(dataset_size))
    if seed:
        np.random.seed(seed)
    # shuffle which images are used as validation
    np.random.shuffle(data_indices) # in-place

    if validation_size == 0:
        print("Validation size is 0!")
        raise
    else:

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
                                               val_batch_size,
                                               sampler=valid_sampler,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    return train_loader, valid_loader



# Cell

def get_cifar10_train_test_dataloaders(batch_size,val_batch_size,
                                       shuffle=True,
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

    data_dir = get_torchvision_data(dataset_name='Cifar10')
    # then copy to slurm
    data_slurm_dir = utils.copy_folder_to_slurm_tmpdir(data_dir)

    print(data_slurm_dir, data_slurm_dir.parent)

    # load training dataset (as we are doing a split)
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset =  torchvision.datasets.CIFAR10(root=data_slurm_dir.parent, train=True,
                                            transform=dataset_transform, download=True)

    test_dataset =  torchvision.datasets.CIFAR10(root=data_slurm_dir.parent, train=False,
                                            transform=dataset_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(test_dataset, val_batch_size,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory)

    return train_loader, valid_loader