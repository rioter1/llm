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
    


    
class MLP(nn.Module):

    # MLP class works like an encoder decoder class where th input is 4xd and later cut
    # by 4x
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):

    # defining an attention block which consists of applying 
    # attention on normlaized input
    # followed by MLP
    # this block structure is replecated multiple times (96 for chatgpt)

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

# dataclass is only used for storing data 
# without defining any boiler plate code


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # wte and wpe are token and positional embeddings
        # Token embeddings are used to convert discrete 
        # tokens (e.g., words, subwords, or characters) into dense vector representations.
        # block size is the maximum length of the sequence or the timesteps needed to look forward
        # vocab size is the max number of words present in the vocabulary
        # ModuleDict allows you to easily access components by name, 
        # while ModuleList allows you to iterate over layers or blocks


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight 

        # normal initialization fro weights and embedding and zero intialization for bias
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # init all weights
        # The apply method is a built-in method of nn.Module 
        # in PyTorch. When you call self.apply(self._init_weights), 
        # PyTorch automatically traverses all submodules (layers) 
        # of the model and applies the provided function (_init_weights)
        # to each of them.

        # he named_parameters() method 
        # returns an iterator over the model's parameters
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        # This is a common technique to scale the initialization based on the number of layers, 
        # which can help stabilize training.projection layers in transformers) may require a different scale of initialization to ensure effective learning.
        # This is particularly relevant in deep networks, 
        # where the depth can lead to issues like vanishing or 
        # exploding gradients. Adjusting the initialization 
        # for specific layers can help mitigate these issues.

    def get_num_params(self, non_embedding=True):
        # You can call self.parameters() on any subclass of 
        # nn.Module, and it will provide 
        # you with an iterator over all parameters in the model
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # If non_embedding is true, the positional embedding is subtracted
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
