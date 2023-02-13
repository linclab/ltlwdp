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

from .base_models import Flatten, Dropout, Network, MaxPool

class Conv_cSGD_Mixin():
    "DANN update corrections for convolutional network"

    def __init__(self, csgd_inplace=False, **kwargs):
        self.csgd_inplace = csgd_inplace
        super().__init__(**kwargs)

    def update(self, layer, **kwargs):
        """
        This is the same as the MLP corrections however different dims
             - ne is output channels of (econv)
             - n_input, or d is from kernel size etc
        """
        d  = layer.d
        ne = layer.e_conv.out_channels

        with torch.no_grad():
            if self.csgd_inplace:
                layer.i_conv.weight.grad.mul_(1/ np.sqrt(ne))
                layer.Wei.grad.mul_(1/ d)
                layer.alpha.grad.mul_(1/ (np.sqrt(ne)* d))
            else:
                layer.i_conv.weight.grad = layer.i_conv.weight.grad / np.sqrt(ne)
                layer.Wei.grad =  layer.Wei.grad / d
                layer.alpha.grad =  layer.alpha.grad / (np.sqrt(ne)*d)
        super().update(layer, **kwargs)

class DalesCNN_SGD_UpdatePolicy(BaseUpdatePolicy):
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g.
    Here Wex, Wix are the e and i convolution filter banks respectively.

    This should be inherited on the furthest right for correct MRO.

    BaseUpdatePolicy just inherits nn.Module at the moment.
    '''
    def update(self, layer, **args):
        """
        Args:
            lr : learning rate
        """
        lr = args['lr']
        with torch.no_grad():
            for key, p in layer.named_parameters():
                if p.requires_grad:
                    p -= p.grad * lr

            layer.i_conv.weight.data = torch.clamp(layer.i_conv.weight, min=0)
            layer.e_conv.weight.data = torch.clamp(layer.e_conv.weight, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            layer.g.data   = torch.clamp(layer.g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()


class DalesANN_conv_cSGD_UpdatePolicy(Conv_cSGD_Mixin, DalesCNN_SGD_UpdatePolicy):
    pass


# Cell
class EiConv_WeightInitPolicy(BaseWeightInitPolicy):

    def init_weights(self, layer):

        # alpha same as before apart from d is defined differently
        a_numpy = np.sqrt((2*np.pi-1)/ (layer.d)) * np.ones(shape=layer.alpha.shape)
        a = torch.from_numpy(a_numpy)
        alpha_val = torch.log(a)
        layer.alpha.data = alpha_val.float()

        # init E and I filter weights
        # for MLP hidden target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))

        target_std = np.sqrt(2*np.pi/ (layer.d*(2*np.pi-1)))
        exp_scale = target_std # The scale parameter, \beta = 1/\lambda = std
        Wex_np = np.random.exponential(scale=exp_scale, size=(layer.e_conv.weight.shape))

        if layer.i_conv.out_channels == 1:
            Wix_np = Wex_np.mean(axis=0, keepdims=True) # not random as only one int
            Wei_np = np.ones(shape = layer.Wei.shape)/layer.i_conv.out_channels

        elif layer.i_conv.out_channels != 1:
            # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
            Wix_np = np.random.exponential(scale=exp_scale, size=(layer.i_conv.weight.shape))
            Wei_np = np.ones(shape = layer.Wei.shape)/layer.i_conv.out_channels

        layer.e_conv.weight.data = torch.from_numpy(Wex_np).float()
        layer.i_conv.weight.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()
        nn.init.zeros_(layer.b)
        nn.init.ones_(layer.g)


class EiConvLayer(nn.Module):

    def __init__(self, in_channels, e_channels, i_channels, e_kernel_size, i_kernel_size,
                 nonlinearity = F.relu, update_policy=DalesANN_conv_cSGD_UpdatePolicy(),
                 weight_init_policy = EiConv_WeightInitPolicy(),
                 e_param_dict = {'stride':1, 'padding':0, 'dilation':1, 'groups':1, 'bias':False, 'padding_mode':'zeros'},
                 i_param_dict = None, learn_gain_bias=True):
        """
        Args:

            i_param_dict : if None, inherits the e_param dict. For now this should be None
            learn_gain_bias : Whether to learn the gain and bias of the normalisation operation
                            Expected that learn_gain_bias Truem and the e/i convs have no bias 
                            note at the moment learn_gain_bias does nothing!

        """
        super().__init__()
        self.nonlinearity = nonlinearity

        # Fisher corrections are only correct for same e and i filter params
        # Therefore set i_params to e_params
        if i_param_dict is not None: raise
        else: i_param_dict = e_param_dict

        if e_param_dict['bias'] == False and learn_gain_bias == False:
            print("Warning, are you sure you want both conv bias and gain bias False?")
        if e_param_dict['bias'] and learn_gain_bias:
            print("Warning, are you sure you want both conv bias and gain bias True?")

        if e_param_dict['bias']:
            print('For not you were intending to not learn conv biases?')
            raise

        self.e_conv = nn.Conv2d(in_channels, e_channels, e_kernel_size, **e_param_dict)
        self.i_conv = nn.Conv2d(in_channels, i_channels, i_kernel_size, **i_param_dict)

        # inhibitory to excitatory weights for each output activation map
        self.Wei = nn.Parameter(torch.randn(e_channels, i_channels))
        self.alpha = nn.Parameter(torch.ones(size=(i_channels, 1, 1)), requires_grad=True)

        self.epsilon = 1e-8 # for adding to gamma_map

        # one gain and bias for each filter
        self.g = nn.Parameter(torch.ones(e_channels, 1,1))
        self.b = nn.Parameter(torch.zeros(e_channels, 1,1))

        self.update_policy = update_policy
        self.weight_init_policy = weight_init_policy

        # assign d (fan_in) for weight init etc
        self.d = int(np.prod(self.e_conv.weight.shape[1:])) # shape is out_c, in_c, kernel W, kernel H

    def forward(self, x):
        self.e_act_map = self.e_conv(x)
        self.i_act_map = self.i_conv(x)

        # produce subtractive map
        self.subtractive_map = (self.Wei @ self.i_act_map.permute(2,3,1,0)).permute(3,2,0,1)

        # produce a divisve map
        self.gamma = self.Wei @ (torch.exp(self.alpha) * self.i_act_map).permute(2,3,1,0)
        self.gamma = self.gamma.permute(3,2,0,1) + self.epsilon

        self.zhat = self.e_act_map - self.subtractive_map
        self.z = (1/ self.gamma) * self.zhat

        self.z = self.g*self.z + self.b

        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z
        return self.h

    def update(self, **kwargs):
        self.update_policy.update(self,**kwargs)

    def init_weights(self, **kwargs):
        self.weight_init_policy.init_weights(self, **kwargs)

    def extra_repr(self):
        return "Nonlinearity: "+str(self.nonlinearity.__name__)

    def __repr__(self):
        """
        Here we are hijacking torch from printing details
        of the weight init policies

        You should make two reprs , one to print these detaisl
        """
        return f'e{self.e_conv.__repr__()} \n     i{self.e_conv.__repr__()}'