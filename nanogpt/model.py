import math
import inspect 
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    Layernormalization is a method where all normlaization of a matrix happens column wise. 
    for example in the attention matrix, the output is normlaized column wise
    """
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(ndim))
    
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
