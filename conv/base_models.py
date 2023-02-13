
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import ariamis.utils.all_utils as utils
from ..utils.all_utils import Progbar, acc_func, in_notebook

from ..data import mnist, cifar

from ..dense.base_models import BaseWeightInitPolicy
from ..dense.base_models import DenseLayer, DenseNet, build_densenet, SGD
from ..dense.base_models import LayerNormLayer, BatchNormLayer

class Flatten(nn.Module):
    """Flattens all but the batch dimension"""
    def __init__(self, input_shape=None):
        """
        Args:
            input_shape: Shape of each batch element, ie. x.shape[1:]
                         optional, used for n_output property.
        """
        super().__init__()

        self.input_shape = input_shape
        self.network_index = None  # this will be set by the Network class obj
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self,x):
        batch_size=x.shape[0]
        return x.reshape(batch_size,-1)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            return np.prod(self.input_shape)

class Dropout(nn.Module):
    """An unneccesary wrap of torch's nn.Dropout"""
    def __init__(self, drop_prob=0.5, input_shape=None):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob, inplace=False)
        self.input_shape = input_shape
    def forward(self, x):
        return self.dropout(x)

    @property
    def output_shape(self):
        if self.input_shape is None:return None
        else: return self.input_shape

class MaxPool(nn.Module):
    """
    Like the conv layer, this is a simple wrapper around
    torch.nn.MaxPool2d
    """
    def __init__(self, kernel_size, stride, padding, input_shape=None):
        """
        Args:
            kernel_size, stride, padding - see nn.MaxPool2d docs
            https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

            input_shape: Shape of each batch element, ie. x.shape[1:]
                         optional, used for n_output property.
        """
        super().__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size, stride, padding)
        self.input_shape = input_shape

        self.network_index = None  # this will be set by the Network class obj
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self, x):
        return self.maxpool2d(x)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            data = torch.rand(self.input_shape).unsqueeze(0)
            return self.forward(data).shape[1:]

    def __repr__(self):
        """
        Here we are hijacking torch from printing details
        in a clunky way (as it views this as being two children)
        """
        return self.maxpool2d.__repr__()

class HeConv2d_WeightInitPolicy(BaseWeightInitPolicy):
    """
    Remember BaseWeightInitPolicy is basically just nn.Module
    """
    @staticmethod
    def init_weights(conv2d):
        """
        Args:
            conv2d - an instance of nn.Conv2d

        a combination of Lecun init (just fan-in)
        and He init (numerator is 2 due to relu).

        References:
        https://arxiv.org/pdf/1502.01852.pdf
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        """
        fan_in = np.prod(conv2d.weight.shape[1:]) # we scale weights for each filter's activation
        target_std = np.sqrt((2 / fan_in))

        if conv2d.bias is not None:
            nn.init.zeros_(conv2d.bias)

        nn.init.normal_(conv2d.weight, mean=0, std=target_std)

class ConvLayer(nn.Module):
    """
    Standard Conv2d

    This is a clunky implementation just wrapping Conv2d for similarity with ei conv layers.
    By defining this way network classes and polcies etc can be shared.
    """
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity = F.relu,
                 weight_init_policy=HeConv2d_WeightInitPolicy(), update_policy=SGD(), input_shape=None,
                 conv2d_kwargs={'bias':True, 'stride':1, 'padding':0, 'dilation':1, 'groups':1,'padding_mode':'zeros'}):

        super().__init__()
        self.nonlinearity = nonlinearity
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, **conv2d_kwargs)
        self.input_shape = input_shape

        self.weight_init_policy = weight_init_policy
        self.update_policy = update_policy
        self.network_index = None  # this will be set by the Network class obj
        self.network_key = None  # the layer's key for network's ModuleDict

    def forward(self, x):
        self.z = self.conv2d(x)
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h

    def update(self, **kwargs):
        self.update_policy.update(self, **kwargs)

    def init_weights(self, **kwargs):
        "Not sure if it is best to code this as be passing self.conv tbh"
        self.weight_init_policy.init_weights(self.conv2d, **kwargs)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None
        else:
            data = torch.rand(self.input_shape).unsqueeze(0)
            return self.forward(data).shape[1:]

    def extra_repr(self):
        return ""

    def __repr__(self):
        """
        Here we are hijacking torch from printing details
        of the weight init policies
        """

        return self.conv2d.__repr__()

class Network(nn.Module):
    """
    Class to represent network of convolutional and fully connected layers.

    This is slightly more complicated than for dense as some layers will have parameters (and
    have update and weight init policy attributes), whereas others such as pooling
    and flatten layers will not. Layers are expected to implement forward, but optionally
    can implement update and init_weights methods.


    Note: this still expects a straightforward sequential structure.
    """
    def __init__(self, module_list=None):
        """
        Args:
            module_list: an optional list of modules to be added to self.layers
        """
        super().__init__()
        self.initialised = False # tracks if weights have been initialised

        self.layers = nn.ModuleDict()  # only contains layers with parameters
        for i, module in enumerate(module_list):
            key = ''+str(i)
            self.layers[key] = module
            self.layers[key].network_index = i
            self.layers[key].network_key = key

        self.__dict__.update(self.layers) # enable access with dot notation

    def check_dimensions(self, x, silent=False, layer_details=False):
        """
        Sets the dimensions of the layers and checks random data goes through

        Not sure if this is the best way...
        For e.g how it interacts with the dense layers is hacky.
        """
        if not silent: print('Model input:', x.shape[1:])
        for i, (key, layer) in enumerate(self.layers.items()):
            if i == 0: self[0].input_shape = x.shape[1:]
            # these shapes are not including batch dimension
            if not silent:
                print(key,layer.output_shape, layer.__class__.__name__, )
                if layer_details: print("  ", layer)
            if i < len(self.layers)-1:
                try:
                    self[i+1].input_shape = layer.output_shape
                except AttributeError:
                    # for denselayers input_shape is a property
                    assert self[i+1].input_shape == layer.output_shape


    def append(self, layer, key=None):
        """
        Appends layer to the network.

        Not implemented yet, we will require that the layer has
        a n_output.attr that is not None.
        """
        pass

    def forward(self, x):
        for key, layer in self.layers.items():
            x = layer.forward(x)
        return x

    def update(self, **args):
        for key, layer in self.layers.items():
            if hasattr(layer, "update"): layer.update(**args)

    def init_weights(self, **args):
        for key, layer in self.layers.items():
            if hasattr(layer, "init_weights"): layer.init_weights(**args)
        self.initialised = True

    def extra_repr(self):
        return ''

    def __getitem__(self, item):
        "Enables layers to be indexed"
        if isinstance(item, slice):
            print("Slicing not supported yet")
            raise
        key = list(self.layers)[item]
        return self.layers[key]