To run training 

```
cd nanogpt
python train.py config/train_shakespeare_char.py
```
result: iter 5000: loss 0.8205, time 9107.82ms, mfu 24.47%

Run
```
python sample.py --out_dir=out-shakespeare-char
```
OUTPUT

'''

 
Overriding: out_dir = out-shakespeare-char
number of parameters: 10.65M
Loading meta from data/shakespeare_char/meta.pkl...


Clown:
So, we will be longest
back on the danger of a man again
here that excused here? we thank ye away, what
for a wizard with the toft, and that where it beg it
extrement the sea, to be patient now.

AUTOLYCUS:
A ling of mine eye on him.

AUTOLYCUS:
Why, my lord, I am patiently to thy woe,
And I am the one that is so it is, this a gentleman poor
A shoulderer, benter for such a great whiteful as of the
pedlar?

Provost:
This is a better:
Are you he a propheciest to a save your tongue, your
wi
---------------

Men pleasure, sirrah, I would have had not some noble two
better to the subjects of the court.

JULIET:
I may say you love you so your executioner,
Perhaps over-time to come the plaint; but it is
Even to resign of your gracious prince,
And with tears of the children to bless but what see you she
Will deny of love? Privy, most doing,
To be dull put on't. Let me throw the other undernoon
In the boar of my hand with that will she were beloved on the king:
So, for this time would have us your wife.

---------------

Messenger, but seal it, I would what have concernited
As it does, withal.

LEONTES:
Me, my son,
She shall be his body for his truth:
Why, now he should goes his father, take the crown,
As he and in his delaying self-sentenced brow not on him.

HERMIONE:
But you had been a herd of mine honour,
Being constraction or our crown blood. Here's be your shame
To be brief on his desire. For she was not caught
With the common about men and her banished
With meeties of the pity of our youth, and wholesome

---------------


FLORIZEL:
He should they will give.

RICHARD:
What a scene but the seers where he were,
But we were all ready of my Lord Norfolk,
The townry of the royal blood,
That I was that becomes his heirs, that which he I
Am confined to the greatest gods; and, he cracks
And I will not, would take my part. Go you as the ground,
Who seems already like a sound fool. The valour of the queen,
Cursed for most home, and sent my and to be gone;
And see how for the winds of the golden king's course,
And will be r
---------------

That lamb did thus this my forefaction light for this gold fash,
To death upon their daughter not our hands.

YORK:
Good lord, lord, we will not be requited
To this is gone crown of her breast.

HASTINGS:
A man of many hours it now,
Where the one subjects of chamber'd them,
To the soldiers, down and drinking when it shall were in his
Than the dismissession shows my son, that the envious voices
The hollow of Lancaster Catesby be a traitor,
Too as dead, and who we will profess of their furthens,
N
---------------


MENENIUS:
Have been so the coming to her to grant in his son
and honours his face, he was wings to my foes
and shall be the wreath of my chamber.

Messenger:
Tell him what what think you his face?

MENENIUS:
The proceeding, how you have been are king, and for your
noise the prayers, or your service, and with no more
advised with city. You, till I be grocented with your virtue. So, if
you do it be so, that you shall have good to myself, that
the gentleman have been death to see a right world spi
---------------

She will be a gracious trages, and shall, when it is not to man
to see me as the place of a shrone: yea, I know why,
therefore thou know'st a drinken to right, I'll have of thee.

BENVOLIO:
I never saw the prince of the court.

ROMEO:
The sour shepherd hath been or no foot, says and
beggar; within the fieldst world and make him worth.

MERCUTIO:
Indeed, rought with madness, I will have a custom of honour,
to be colded by cushion and possession of the creets,--

HERMIONE:
Ay, so I have done, sir,
---------------

late that doth sit not for every mind; and therefore,
I say the which I have rested the man
And the bowels of my life and my noble loving
What may be concluded again. Surped with this
As it is so, when you shall have done you suspice
And first him to your ears as say your honests
Whereby here do our state, as the world were not!
Here comes Romeo to Romeo him,
And to Romeo come thee for hence this executioner!

FRIAR LAURENCE:
Peter! how now, much pain me with one so a wimad?

BENVOLIO:
Thou wilt
---------------


LEONTES:
The Lord Antigonus is gracious servantment,
Because content of vain service!
The prisoners have done thee to slept the house;
And if thou comest the forest be monstary,
Which thou takest where? why, then I speak, if thou wouldst for
Thine own with coal whom they have sing not.
The very heavier is no honour'd, who does as if
Thou gave me there such blood with their guilty hates.

HASTINGS:
Here comes Queen, our love, with money at haste.
What will you distard your son Claudio?

QUEEN EL
---------------

He hath not seen resolved himself and spoil,
If I cannot give my knees from him.

HENRY BOLINGBROKE:
Come, come, go some, sir: by his trith
Is this master than ever a woman's fear,
And virtues her with his own life with his soul,
And all in him with her services and fet
And land-discontented upon her hate,
And in his jewel and beget his despised by him,
And hardless a lower sleeper, before him, and his shape
To see him in his three hopeful sir.
That comes the other hath ever common him
To set hi
---------------

'''

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


