import numpy as np

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import ariamis.utils.all_utils as utils
from ..utils.all_utils import Progbar, acc_func, in_notebook

from ..data import mnist, cifar

from ..dense.base_models import BaseWeightInitPolicy, BaseUpdatePolicy, SGD
from ..dense.base_models import DenseLayer, DenseNet, build_densenet
from ..dense.base_models import LayerNormLayer, BatchNormLayer

from ..dense.ei_layers import EiDenseWithShunt
from ..dense.ei_layers import EiDenseWithShunt_WeightInitPolicy, DalesANN_cSGD_UpdatePolicy

from .base_models import Flatten, Dropout, Network, MaxPool, ConvLayer
from .ei_layers import EiConvLayer, EiConv_WeightInitPolicy
from .ei_layers import DalesANN_conv_cSGD_UpdatePolicy

class AveragePool(nn.Module):
    def __init__(self, input_shape=None):
        """
        Args:
            input_shape: Shape of each batch element, ie. x.shape[1:]
                         optional, used for n_output property.
        """
        super().__init__()

        self.avgpool2d = nn.AdaptiveAvgPool2d((4,4)) # (4,4 is the desired output size h x w)

        self.input_shape = input_shape
        self.network_index = None  # this will be set by the Network class
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self, x):
        return self.avgpool2d(x)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            data = torch.rand(self.input_shape).unsqueeze(0)
            return self.forward(data).shape[1:]

    def __repr__(self):
        """
        Here we are just hijacking torch from printing details
        in a clunky way (as it views this as being two children)
        """
        return self.avgpool2d.__repr__()

# Cell
def vgg_layers(vgg19=False, dropout_prob=0):
    """
    Args:
        vgg19 : bool, return either vgg16 or vgg19.
    """
    kwargs={'stride': 1, 'padding': 1, 'dilation': 1,
            'groups': 1, 'padding_mode': 'zeros', 'bias':True}
    layers = [
            ConvLayer(3,64,3,conv2d_kwargs=kwargs),
            ConvLayer(64,64,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(64,128,3,conv2d_kwargs=kwargs),
            ConvLayer(128,128,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(128,256,3,conv2d_kwargs=kwargs),
            ConvLayer(256,256,3,conv2d_kwargs=kwargs),
            ConvLayer(256,256,3,conv2d_kwargs=kwargs),
            ConvLayer(256,256,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(256,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            ConvLayer(512,512,3,conv2d_kwargs=kwargs),
            MaxPool(2,2,0),
            AveragePool((4,4)),
            Flatten(),
            DenseLayer(512*4*4, 4096, nonlinearity=F.relu),
            Dropout(drop_prob=dropout_prob),
            DenseLayer(4096, 4096, nonlinearity=F.relu),
            Dropout(drop_prob=dropout_prob),
            DenseLayer(4096, 10, nonlinearity=None),
        ]

    if vgg19:
        return layers

    # Below convert vgg19 to vgg16 
    # we need to remove the last conv layer in the last three filterbanks
    # (which we locate using the max pool layers)
    to_remove = []
    for index,l in enumerate(layers):
        if isinstance(l,MaxPool): to_remove.append(index-1)
    to_remove = to_remove[-3:] # only remove the last three convs
    for conv_i in to_remove[::-1]: # index from reverse to not change indexes
        del layers[conv_i]
    return layers

# Cell
def ei_vgg_layers(vgg19=False,dropout_prob=0):
    """
    Args:
        vgg19 : bool, return either vgg16 or vgg19.
    """
    conv2d_kwargs={'stride': 1, 'padding': 1, 'dilation': 1,
                   'groups': 1, 'padding_mode': 'zeros', 'bias':False}

    layers = [
            EiConvLayer(3,64,10,3,3, e_param_dict=conv2d_kwargs),
            EiConvLayer(64,64,10,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(64,128,20,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(128,128,20,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(128,256,40,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(256,256,40,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(256,256,40,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(256,256,40,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(256,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            EiConvLayer(512,512,80,3,3,e_param_dict=conv2d_kwargs),
            MaxPool(2,2,0),
            AveragePool((4,4)),
            Flatten(),
            EiDenseWithShunt(512*4*4, (4096,409), nonlinearity=F.relu,
                             weight_init_policy=EiDenseWithShunt_WeightInitPolicy(),
                             update_policy=DalesANN_cSGD_UpdatePolicy()),
            Dropout(drop_prob=dropout_prob),
            EiDenseWithShunt(4096, (4096,409), nonlinearity=F.relu,
                             weight_init_policy=EiDenseWithShunt_WeightInitPolicy(),
                             update_policy=DalesANN_cSGD_UpdatePolicy()),
            Dropout(drop_prob=dropout_prob),
            EiDenseWithShunt(4096, (10,1), nonlinearity=None,
                             weight_init_policy=EiDenseWithShunt_WeightInitPolicy(),
                             update_policy=DalesANN_cSGD_UpdatePolicy()),
        ]

    if vgg19:
        return layers

    # Below we convert vgg19 to vgg16 
    # we need to remove the last conv layer in the last three filterbanks
    # (which we locate using the max pool layers)
    to_remove = []
    for index,l in enumerate(layers):
        if isinstance(l,MaxPool): to_remove.append(index-1)
    to_remove = to_remove[-3:] 
    for conv_i in to_remove[::-1]: 
        del layers[conv_i]
    return layers