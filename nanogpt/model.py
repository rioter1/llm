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
    rather than using the raw normalized values
    
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

        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        # output projection
        self.c_prof = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("Warning: Using slow attention, flash attention avaliable with ptorch>2.0")
            # here we register the buffer tensor of shape (1,1,block_size, block_size) as a parameter of 
            # pytorch module. a buffer is a pytorch module that is not updated during the training process
            # the name of he buffer here is "bias". the purpose of this buffer is attention masking
            # which help the current token to not get infuned by future tokens
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            
        
    def forward(self, x):

        B,T,C = x.size() # batch size, sequence length and embedded dimensionality

        # The output of the linear layer is a single tensor with shape (batch_size, sequence_length, 3 * n_embd)
        # split is happening along the dimension 2 i.e. the last dimension
        # Sequence lenght of T is the number or token or the timesteps in the input sequence
        # and C is the number of input features i.e. n_embd
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)

        # so if x is 32,10,128 dim tensor, the c_attn function will be applied along the last diemnsion
        # i.e. n_embed and convert it to 3*n_embd
        # the values in the B and T dimensions do not change, this is effectively a 
        # last layer conversion op with x to 3x dimension conversion.
        # this is just how the linear layer works, for a 2 dim, B, C it will only operate on the C dimension
        # also it is NCHW or BCHW convention in torch

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # View function is used to reshape the tensor. 
        # we do this for creating inputs for multiple input heads


        # Calculate attention matrix (softmax(q.k(T)/sqrt(n_embd))).v
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # For each batch and each attention head, the attention scores tensor 
        # att (shape (T, T)) is multiplied by the value tensor v (shape (T, hs)).



        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y