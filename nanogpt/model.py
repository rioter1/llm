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
    
    Layernorm for a simple implemetation would only need 1 argument i.e. input
    but here the the input_shape is needed to determine the correctness
    of dimension, as in LLMs the input size is varying and this helps
    with working with variable size matrices

    the weights and bias are needed to learn a more flexible affine transformation
    rather than using the 
    
    """
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(ndim))
    
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_embd == 0

        # key, query, value 