To run training 

```
cd nanogpt
python train.py config/train_shakespeare_char.py
```
result: iter 5000: loss 0.8205, time 9107.82ms, mfu 24.47%

Notes on GPT(Karapthy)  

GPT2 124M model  
state_dcit= Raw tensors  
token embedding = [50257,768] each token is a 768 dim embedding  
position embedding = [1024,768] max sequence length is 1024, 1024 positions need attending from the past  
psitional embeddings have structure. in range from 0 to 1024, each row is the representation of that postion. each row learns sines and cosines associated with positions  
the sines and cosines are smooth for a well trained model  
in gpt2 positional embeddings are parameters and not sinusoids  

GPT2 is a decoder only transformer, hence cross attention is also missing  
layer norm was added and positions changed  
skeleton of gpt2  
submodule inside a transformer object = token embedding+positional embedding  + num_layer*blocks + layer_norm + linear layer  

A clean residual pathway helps propogate gradient back to the input   
it can have a pre normalization version or post, where 
Residual pathways haveing normalizatio inside them is not good or desirable, gpt2 is a prenormalization version  
mlp happens with every single token indivdually, there is no infrmation exchange between tokens whereas in attention, information is exchanged between the 1024 tokens  

gelu non linearity is like relu but no flat tail at exactly 0  
gelu always gives a local gradient in comparison to relu which makes it 0  
multi head = concatenated multiple heads of attention, uses a modulelist of multile head objects  

each token emits 3 vectors (QKV)  
number of token = 50257 = 50000 BPE merges + 256 bytes tokens +1 End of text   
attn.bias is a buffer which is used for autoregressive mask hence it can be ignored when copying keys from the hugging face model to your own transformer model  

input indices are always of shape (B,T) where B is batch dimension and we have the time dimension  
therefore B independent sequences of T sequence length  
position embedding (T, n_embed), positional embeddings are going to be identical for every single row and so there is broadcasting hidden inside any + operation with PE  
token embedding  (B,T,n_embed)  
input = TE + PE  
forward pass output logits  
when not training the model but only using it, put the model in eval model using model.eval()  

encoding = string becomes a list of integers  
these encoding are replicated num_return_sequence times(BATCH B)  
therefore intial input becomes num_return_sequnces,token lentgh(TIME T)  
num_return_sequences = number of returned sequences you expect decoder to give for 1 input sentence  

1 more value needs to added to EVERY ROW i.e. 1 more additional column which are the logits  
The logtis only at last columns are important, rest are thrown away  

Here topk is 50, top 50 probabilities taken for tokens, rest made to 0  

The columns in X(input) grow with every loop iteration i.e. with every loop iteration 1 logit clumn gets added  

----------------------------------------------------------------------


Training  
tiny shakespeare dataset is the best for debugging  

gpt2 has a compression ratio of 3 to 1 so 1000 charachters are about 300 tokens  

----------------------------------------------------------------------

Gradient accumulation: Instead of updating the model's weights after processing each individual batch of training data, the gradients are accumulated over multiple batches before updating.gradients are summed up over multiple batches rather than immediately incorporating the information from a single batch into the model's parameters  
Once a certain number of batches have been processed, the accumulated gradients are used to update the model parameters using an optimization algorithm like SGD or Adam.  
The effective batch size is the product of the actual batch size and the number of accumulation steps. For example, if the micro-batch size is 1 and gradient_accumulation_steps is 8, the effective batch size is 8.  
1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter  

----------------------------------------------------
How to calculae the number of parameters for a transformer model assuming only 1 transformer layer ?

if the embedding size is d_model and the number of heads is n_heads, then the dimensions of each Q, K, and V weight matrix would typically be:
d_model x d_k (k=key)
d_model = n_embd (for example in our case 768)
d_k = d_model/num_head (for 12 heads = 64)

there for 3 q,k,v matrix per layer = 3x768x64
Total weights for Q, K, V projections = (768 × 64 × 3) × 12 = 147,456

The dimensions of the output projection matrix are: (d_model × d_model) = (768 × 768)
589,824
In addition to q,k,v matrices, each layer has 2 Feed forward networks with a relu activation inbetween, a layer normalization and a residual conncetion which does not have a weight but is part of the network

The dimensions of the first linear layer are: (d_model × 4d_model) = (768 × 3072) = 2,359,296

The dimensions of the second linear layer are: (4d_model × d_model) = (3072 × 768) = 2,359,296

Total Weights = Attention Weights + Feedforward Weights
= 737,280 + 4,718,592
= 5,455,872

Embedding Parameters=V×d_model
For a vocabulary size of 50,304 and an embedding size of 768 768×50,304=38,707,712

The output layer typically projects the hidden states back to the vocabulary size to produce logits for each token in the vocabulary​
768×50,304=38,707,712


Total Parameters=5,460,000+38,707,712+38,707,712 
Approx 82 million


so for gpt2 with 12 layers, we have 5.46x12 ~ 65 million + 76 million which comes out roughly 130-140 million


