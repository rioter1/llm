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

1 more value needs to added to EVERY ROW i.e. 1 more additional column which are the logits  
The logtis only at last columns are important, rest are thrown away  

Here topk is 50, top 50 probabilities taken for tokens, rest made to 0  

The columns in X(input) grow with every loop iteration i.e. with every loop iteration 1 logit clumn gets added  









