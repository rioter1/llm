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
