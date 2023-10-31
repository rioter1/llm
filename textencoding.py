# Encoding text data into sequences

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example text data
data = [
    "I enjoy answering questions.",
    "How can I help you?",
    "What is your favorite color?",
    "ChatGPT is a language model.",
]

# Step 1: Tokenization
# Tokenization converts text into individual words or subword tokens.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# Vocabulary mapping
word_index = tokenizer.word_index

# Step 2: Text to Sequences
# Convert text sentences to sequences of integers using the tokenizer
sequences = tokenizer.texts_to_sequences(data)

# Step 3: Padding
# Neural networks require input sequences of the same length.
# We pad sequences with zeros to make them of equal length.
padded_data = pad_sequences(sequences)

# Let's see the results
print("Word Index:")
print(word_index)
print("\nSequences:")
print(sequences)
print("\nPadded Data:")
print(padded_data)

# Dimension of the embedding vectors
embedding_dim = 32

# Step 1: Create an Embedding Layer
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=padded_data.shape[1])
embedded_data = embedding_layer(padded_data)

# Step 2: Create Queries, Keys, and Values
Q = tf.keras.layers.Dense(embedding_dim)(embedded_data)  # Queries
K = tf.keras.layers.Dense(embedding_dim)(embedded_data)  # Keys
V = tf.keras.layers.Dense(embedding_dim)(embedded_data)  # Values

# Now, you have derived queries (Q), keys (K), and values (V) from the encoded data.

# Calculate raw attention scores
attention_scores = tf.matmul(Q, K, transpose_b=True)  # Dot product between Q and K
attention_scores = attention_scores / tf.math.sqrt(embedding_dim)  # Scale by square root of the dimension

# Apply the softmax function to get attention weights
attention_weights = tf.nn.softmax(attention_scores, axis=-1)

# Assuming you have attention_weights and values (V) as computed in the previous example

# Compute the context vectors
context_vectors = tf.matmul(attention_weights, V)

# Assuming you have context_vectors and an output layer

# Generate predictions (e.g., the next word in a translation)
# output_layer can be a linear or nonlinear activation layer

predictions = output_layer(context_vectors)


# Define the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Calculate the loss
loss = loss_object(target_sequences, predictions)

# Optionally, calculate the mean loss over all sequences in a batch
mean_loss = tf.reduce_mean(loss)

