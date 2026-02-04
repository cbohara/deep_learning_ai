# [Attention in Transformers: Concepts and Code in PyTorch](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch)
## Intro

Before using Word2Vec
- Static embeddings
- A single word always maps to the same vector
- No context awareness
- Like a dictionary lookup

2015 using RNNs
- Encoder
    - Earlier approaches = single dense vector represented the meaning of the entire sentence
    - Now each word gets its own vector that understands the meaning of the word in the context of the sentence = contextual embedding
    - Contextual embedding = meaning of the word isn't just the word itself, but also on the words around it, on the context  
- Encoder > Decoder 
    - Encoder passes vector per word
    - Decoder weights or attends to the inputs based on the previous and current word being generated

2017 Attention is all you need
- Introduce the transformer architecture
- Highly scalable using GPUs
- Encoder - creates contextual embeddings for the input sentence in a single pass
- Decoder
    - produces the output one word at a time
    - decoder output for previous word passed as input for next word, so it has the context of what just was outputted

Decoder basis for GPT 
- Used for ChatGPT
- And all other popular models 

Encoder basis for BERT (bidirectional encoder representations from transformers)
- contextual embeddings = a single word can get different vectors in different contexts
    - ex: "I went to the bank" (financial) vs "I sat by the bank" (river)
    - The vector depends on the surrounding words

How BERT works
1. base layer = token-to-embedding
- each token gets
    - default embedding vector from a static lookup table
        - no context awareness yet 
        - just uses a default
    - default positional encoding vector
- so we have a vector (list of numbers) that represent both the word and its position in the sentence
```python
# Input: "I went to the bank"
tokens = ["I", "went", "to", "the", "bank"]

# Static lookup (same for "bank" every time, regardless of context)
static_embeddings = {
    "I": [0.1, -0.2, 0.3, ...],      # 768 numbers
    "went": [0.4, 0.1, -0.3, ...],
    "to": [-0.1, 0.2, 0.1, ...],
    "the": [0.0, 0.0, 0.0, ...],
    "bank": [0.5, -0.4, 0.2, ...]    # Same vector every time at this stage
}

# Learned lookup table (simplified)
learned_positions = {
    0: [0.12, -0.05, 0.33, ...],  # Position 0 pattern
    1: [-0.08, 0.21, -0.14, ...], # Position 1 pattern
    2: [0.15, 0.07, 0.28, ...],   # Position 2 pattern
    3: [-0.11, -0.19, 0.09, ...], # Position 3 pattern
    4: [0.06, 0.24, -0.17, ...]   # Position 4 pattern
}
```
2. transformer layer
- modify the default embedding based on the context
- use self-attn to look at all the words in the sentence
- each word's embedding gets updated based on its neighbords
- iterate through multiple layers that handle multiple flavors / variations of language
3. output layer
- finalized vector based on all the self-attn layers 

BERT TL;DR 
- get a vector for each word
- get a vector for each position
- add them together

## Main ideas behind transformers and attention

Transformers = 3 main parts 
- Word embedding - Converts tokens (words / bits of words / symbols) into numbers
- Positional encoding - Helps keep track of word order
- Attention - Establishes relationships between words
    - Ex: Pizza came out of oven and it tastes good = need to map it to pizza, not oven
    - Self-attention = see how similar each word is to the other words in the sentence, including itself
    - Calculates similarity score between each word to each other


## Matrix math for calc self-attention

Attention(Q, K, V) = SoftMax
- Q = query - search question - I'm looking for the last name closest to Stammer in my list of reservations
- K = key - last names available to search
- V = value - value that maps to key most appropriate for the query

```python
hotel_database = {
    "Strarmer": 919,
    "Starmer": 537,
    "Summer": 214
}
```

1. Tokenization = word > token
    - Each word typically becomes 1 token
    - "write a poem" = 3 tokens: ["write", "a", "poem"]
2. Embedding = token > vector
    - Each token becomes 1 vector with many dimensions
    - "write" > 1 vector (aka list) with 768 dimensions (aka numbers) when using BERT

Note
- simplified example - we are using 1 number for word embedding + 1 number for positional encoding 
- realistically using vector (list of numbers) and adding element wise (aka adding index 0 of one list to index 0 of another list, etc)

Prompt - "write a poem"
```python
import numpy as np 

# encoded values 
encoding_value = np.array([
    [1.16, 0.23], # "write": word=1.16 + position=0.23 = 1.39 (if added)
    [0.57, 1.36], # "a": word=0.57 + position=1.36 = 1.93 (if added)
    [4.41, -2.16] # "poem": word=4.41 + position=-2.16 = 2.25 (if added)
])
```

Query weight values
- learned so the model ask useful questions about the relationship between words
- they don't represent fixed meaning 
- they are learned

Think of parameters as a knob on a radio
- Each knob controls one aspect 
    - Volume knob = param1
    - Bass knob = param2
    - Treble knob = param3
- You adjust the knobs to get the desired sound change
- In ML - the model has millions of knobs (parameters) to adjust to get the desired prediction change 

Big picture in ML the parameter is 
- A number the model learns
- Stored in the model
- Used in calculations
- Updated during training 

BERT has ~110 million parameters
- like a spreadsheet with ~110 million cells
- each cell holds 1 number (a parameter)
- training fills in the best numbers
- inference uses those numbers to make predictdions

Prompt - "write a poem"
```python
import numpy as np 

# encoded values 
encoding_value = np.array([
    [1.16, 0.23], # "write": word=1.16 + position=0.23 = 1.39
    [0.57, 1.36], # "a": word=0.57 + position=1.36 = 1.93
    [4.41, -2.16] # "poem": word=4.41 + position=-2.16 = 2.25
])

query_weights = np.array([
    [0.54, -0.17],
    [0.59, 0.65]
])

Q = np.array([
    [0.76, -0.05], # write
    [1.11, 0.79], # a
    [1.11, -2.15] # poem
])

key_weights = np.array([
    [0.54, -0.17],
    [0.59, 0.65]
])

K = np.array([
    [0.76, -0.05], # write
    [1.11, 0.79], # a
    [1.11, -2.15] # poem
])

value_weights = np.array([
    [0.54, -0.17],
    [0.59, 0.65]
])

V = np.array([
    [0.76, -0.05], # write
    [1.11, 0.79], # a
    [1.11, -2.15] # poem
])
```

Different weights serve different roles
- Query = what to look for = how to ask good questions
- Key = how to identify matches = how to find good identifiers or tags
- Value = what info to retrieve = how to extract useful info