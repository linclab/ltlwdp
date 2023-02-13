
__all__ = ['acc_func']

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def acc_func(yhat,y):
    max_vals, arg_maxs = torch.max(yhat.data, dim=1)
    # arg_maxs is tensor of indices [0, 1, 0, 2, 1, 1 . . ]
    total   = yhat.size(0)
    correct = (arg_maxs == y).sum().item()
    return correct/total