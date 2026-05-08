# [Deep Learning Specialization](https://learn.deeplearning.ai/specializations/deep-learning/information#course-outline)

# Course 1

## What is a neural network?
Non-linear function
- graph isn't a straight line
- the relationship changes at different points
- ReLU = rectified linear unit
    - negatives -> 0
    - positives -> unchanges 
    - avoids negative numbers
    - ex: housing price prediction doesn't go below zero
- a neuron computes a linear function z = Wx + b followed by an activation function

Example - housing price prediction
- size + # bedrooms -> neuron -> predicts family size                    ->
- zip code -> neuron -> predicts walkability                             -> price 
- zip code + wealth of inhabitants -> neuron -> predicts school quality  -> 

Magic of neural networks
- all you need to give is the inputs and the outputs as training data 
    - x = size, number bedrooms, zip code, wealth
    - y = price 
- it will figure out everything in the middle 
    - not super concrete what the middle hidden units actually describe
    - hidden units take in all features as input, not just certain ones
- generally performs best with supervised learning 

## Supervised learning with neural networks
Standard neural network
- Follows basic design described above
- Ex: Home features -> price

CNN = convolution neural network
- Typically used for images
- Ex: tagging an object in an image

RNN = recurring neural network
- Used for sequence data (ie need to know the order of data)
- Ex: speech recognition = audio -> text transcript 

## Basics of neural network programing 
Notation
- single training example (x,y)
    - x feature vector
    - y label 
- m training examples
    - (x1, y1), (x2, y2), ..., (xm,ym)

Importance on prediction = feature value * weight
- big weight = feature has a strong influence on prediction
- small weight = feature has little influence on prediction

## Logistic regression
- used for binary classification
- y = sigmoid(w transpose x + b) aka y = sigmoid(z)
    - w transpose x = weighted sum of the inputs
        - one weight per feature
        - model learns the weight of each feature
    - b = bias that shifts the entire prediction up and down
        - one bias value per neuron
        - like y = mx + b where b is the y intercept 
    - pass value into sigmoid function so the output is either 0 or 1
- w and b = parameters the neural network learns
    - to train these parameters, we need to define a cost function

## Model evaluation
- cost function
    - measures how wrong the predictions are
    - high cost = bad prediction
    - low cost = good prediction
- minimize the cost using gradient descent
    - start by setting w and b to any random number
    - calculate slope for the given w and b
    - determine how to adjust w and b to move "downhill" fastest
    - iterate adjusting w and b until reach smallest possible cost = global optima
    - global optima is the w and b where we get the best possible predictions

Graphs
- prediction
    - neural network gives a formula you can plot
    - for a given input x, get prediction y
- graident descent
    - x = a parameter (w or b)
    - y = cost = average loss on the training set
    - shows how wrong the model is for different parameter values
    - use the slope of this curve to update w and b

## Vectorization
- enables us to train on large data sets without using for loops
- improved parallelism
```python
import numpy as np
# for each input feature x, multiply by the weight w
z = np.dot(w, x) + b
```

Review
- a neuron computes a linear function (z = Wx + b)
    - take each feature (x), multiply it by its corresponding weight (W), then sum them
    - bias(b) is the learned offset that shifts the prediction output
- followed by an activation function
    - sigmoid typically used for binary classification (0 or 1 prediction)
    - softmax for multi-class classification
    - ReLU for linear regression to predict a number we don't want to go below 0

- in a neural network
    - each neuron applies a simple nonlinear “bend” to the input space
    - when you compose many of these layers, the bends stack
    - the result is a highly complex decision surface

C1W2_Assignment_2 notes
- overview
    - cat vs non-cat images
    - the model is a single "neuron"
    - takes the pixels, multiplies them by learned weights, adds a bias, then squashes the results to a number between 0 and 1 (probability of cat)
    - the model learns the weights and bias that will get the probabilities closer to the true label   
    - inputs (pixels) -> learn forumla -> probability -> decision

- single training step 
    1. forward propogation - get predictions + cost
        - uses the current weights + bias to predict cat or not for every example and then computes a single number (the cost) that says how wrong the predictions are overall 
            - big cost = doing badly
        - small cost = doing well
    2. backward propogation - get gradients
         - figures out in which direction to nudge the weights and bias so the cost will go down
         - computes gradients = how much each parameter is responsible for the error 
    3. update - change w and b based on gradients
    4. repeat for many iterations 

- summary
    - initialize parameters (weights and biases)
    - optimize the loss iteratively to learn the parameters (weights and biases)
        - compute the cost (is it perdicting well)
        - compute the gradient (how to improve)
        - update the parameters based on the gradient using gradient descent 
    - use the learned parameter to predict cat vs not

## Neural network overview
Review
- input features x and parameters w and b -> "raw score" z
- feed "raw score" z into the activation function -> probability a
- probability a -> prediction based on the value of a 
- calculate the loss L to evaluate the quality of the prediction
- ex: cat 1, non cat 0 - if a > 0.5 -> predict 1 cat, else predict 0 non-cat
- ultimately we repeat the steps above over and over again in a neural network

Layers in a neural network
- input layer - features x
- hidden layer - everything in between
- output layer - provides prediction value (y hat)

each node in network involves 2 computations
1. input features x and parameters w and b -> "raw score" z   
2. pass "raw score" z into activation function -> probability a

layer 1 - input X (features) -> compute Z1 -> apply activation -> get A1 
layer 2 - input A1 (not X)   -> compute Z2 -> apply activation -> get A2 
...
layer n - input An-1 -> compute Zn-1 -> apply activation -> final A = output probability 

# C1W3 refresher 
X = input data 
- each column is 1 example
- each row is a feature (x and y coordinate)
- contains 400 examples
Y = label data 
- for each row, what is the prediciton

```bash
  NumPy (columns = examples):
           point1  point2  point3  ...                                                               
  row 1:    0.5    -0.3     1.1        ← x_coords
  row 2:    1.2     0.8    -0.4        ← y_coords  
```

matrix math
- typical to have each column be a separate example
- can process all columns at once - not one at at time
- instead of looping
```python
for i in range(400):
    z = W1 @ X[:, i] + b1
```
- we can do everything all at once using vectorization
```python
Z1 = W1 @ X + b1
```
- numpy handles the parallelism under the hood

weights and biases don't map to number of examples   
each layer gets one W and one b   

weights and biases define the neural network   
their shape must line up with whatever is fed into that layer   

each hidden neuron computes its own weighted sum and bias

one weight per feature per neuron
- so with 2 feaures, each neuron gets 2 weights  
- neuron needs one weight for each input it receives so it can control how much it "cares" about each feature
- neurons specialize in detecting different patterns during training by determining different weights 

one bias per neuron
- shifts the output up or down

algebra: y = mx + b
neural : z = wx + b
- m / w = slope, weight
- b = y intercept, bias
- x = input feautre

activation function (tanh, sigmoid) squishes the line into a curve   
- from linear line from algebra -> neural network 
- nonlinearity = network learns curves and complex patterns instead of just straight lines

```python
z[1] = w1 * x + b1   
a[1] = tanh(z[1])
```

z1 = raw weighted sum = linear math  
a1 = activation = what you get after squishing z1 thru tanh   

z = multiply + add = raw score   
a = output that gets passed forward  

layer 1 computes 
z1 = w1 * x + b1
a1 = tanh(z1)      

layers 2 recieves a1   
z2 = w2 * a1 + b2
a2 = sigmoid(z2)  

the whole point of the squish is to introduce nonlinearity   
otherwise it's just one big linear equiation   
we only keep the z values separate for backprop step later   

backprop tells u w1 should go down a bit, b2 should go up a bit , etc   
then we adjust parameters (w and b) accordingly    

hidden layer size / width = number of nodes within a single layer = more curves to combine = able to handle more complex pattern    
number of hidden layers / depth = how many layers are stacked between input and output   

bias = neuro's default inclination before any input arives 
- in linear model y = wx + b, the weight w scales the input and b shifts the output up or down regardless of input 
- it biases the prediction towards certain baseline value

general methodology to build a neural network
1. define the neural network structure aka layer size
- size of input layer = number of features in the dataset
- size of the hidden layer = hyperparameter we can set to any value
- size of output layer = number of classes we are trying to predict 
2. initialize the model's parameters 
- weight = how much each input matters
- bias = neuron's default inclination before any input arrives
3. loop
- forward propogation - push the input all the way through the network, layer by layer, to produce a prediction
- compute loss / cost - measure how far the predictions are from the true labels and returns a single number summarizing model performance
- backward propogation - work backwards through the network to compute gradients = a number saying increasing this weight by a tiny amount would change the loss by this much
- update parameters - use gradient descent algorithm to nudge each weight and bias in the direction that reduces cost aka gets closer to the predictions matching ground truth
- repeat until the cost stops decreasing meaningfully (convergence) or you hit a set number of iterations

Input X is almost always a matrix
1. Features per example
- a single training example is typically a vector of features
- ex: house price features [sqft, bedrooms, age, ...]
- so one example x is a vector of length n (# of features)
2. Multiple examples at once
- stack all examples into a table or matrix and process in parallel via vectorization
- vectorization is much faster than looping

hidden layers
- progressively transform raw input into increasingly useful representations
- each transformation re-expresses the data in a form that makes the final decision easier
- the nonlinear activation functions lets the network carve out curved, complex boundaries

output layer
- learned features get collapsed into the actual value we create about
- a probability, a class, a regression value

activation = activation function(weight * inputs + bias)    

debth = # of layers   
width = # of neurons within a given layer   

layers = sequential   
- each layer transforms the previous layer's output
- ex: image recognition
    - layer 1 = edges + color blobs
    - layer 2 = corners, textures, simple shapes
    - layer 3 = object parts - eyes, wheels, leaves (combo of shapes)
    - layer 4 = whole objects - faces, cars, trees
- depth lets the network contruct hierarchical, compositional representations

within a single layer, there are multiple neurons/units/nodes = parallel
- more neurons in a layer = more distinct patterns the layer can recognize in parallel
- a single activation function(weight * input + bias) inside a layer
- each neuron within the layer receives the same inputs (previous layer's activations)
- however each neuron starts with different random weights -> detect different patterns
- produces its own activation

```bash
  Input        Hidden Layer        Output Layer
  (2 features) (4 neurons)         (1 neuron)

    x1 ──┬──► n1 ─┐
         ├──► n2 ─┤
    x2 ──┼──► n3 ─┼──► n_out ──► ŷ
         └──► n4 ─┘
```

GPU demand
- GPUs are in demand because neural network training is overwhelmingly matrix multiplication (matmul)
- GPUs are purpose built to do matmul in parallel
- vs CPU = small number of very powerful cores optimized to do sequential tasks (run general purpose code) fast   
- GPU = thousands of simpler cores that are weaker individually but collectively able to do thousands of simple math operations in parallel 
- training large models is almost entirely matmul
    - GPT 4 scale models involve quintillions of floating point operations per training run
    - on CPU it would take centuries
    - on GPU cluster, weeks
- inference is also matmul
    - every time you send a message to Claude, a forward pass runs through hundreds of billions of weights
    - serving millions of users requires massive GPU capacity
- AI boom created a supply crunch
    - NVIDIA has a moat with GPUs + CUDA
        - PyTorch and Tensorflow are built on CUDA
        - 15+ years of being the default
        - CUDA = software that allows developers to write code for taht hardware without needing to think about low level GPU internals
    - companies are all trying to catch up but NVIDIA has a large head start

parallels to Python vs Pyspark:   
┌──────────────────────┬────────────────────────┬─────────────────────────────┐
│       Concept        │     Python/PySpark     │           CPU/GPU           │                                                                                                             
├──────────────────────┼────────────────────────┼─────────────────────────────┤
│ Sequential engine    │ Python on one machine  │ CPU with few powerful cores │
├──────────────────────┼────────────────────────┼─────────────────────────────┤
│ Parallel engine      │ Spark across a cluster │ GPU with thousands of cores │                                                                                                             
├──────────────────────┼────────────────────────┼─────────────────────────────┤                                                                                                             
│ Abstraction layer    │ PySpark API            │ CUDA / cuDNN                │                                                                                                             
├──────────────────────┼────────────────────────┼─────────────────────────────┤                                                                                                             
│ Underlying workhorse │ JVM + Scala            │ Low-level CUDA kernels      │
├──────────────────────┼────────────────────────┼─────────────────────────────┤                                                                                                             
│ What you write       │ DataFrame operations   │ Tensor operations           │
├──────────────────────┼────────────────────────┼─────────────────────────────┤                                                                                                             
│ What you don't write │ The distribution logic │ The parallelization logic   │
└──────────────────────┴────────────────────────┴─────────────────────────────┘   

## deep neural network
deep = many hidden layers   
- L = # of layers
- n[1] = number of neurons/units/nodes in a layer 1
- a[1] = activation in layer 1
- a[0] = X input feature

forward propogation:
activation[1] = activation function(weight[1] * x input + bias[1]) ->
activation[2] = activation function(weight[2] * activation[1] from previous layer + bias[2]) -> etc etc   

## model release process

a released model is essentially a frozen snapshot of weights  
- after training (which can take weeks to months on huge GPU clusters), the result is a giant file of numbers = billions of weights and biases   
- the same model can be loaded onto thousands of GPUs to serve millions of users in parallel

nuances worth noting
- same model = same weights, but surrounding system can change (ie system prompts, available tools, safety features, etc) without retraining -> different behavior
- models are iteratively updated, but as new releases, not touching previous release
- knowledge cutoff = because the weights are frozen, the model's knowledge is frozen at whatever date its training data ended

every turn, the entire convo is re-fed through the model in entirety 
- 10 messages within a chat, the model doesn't remember the first 9 messages 
- serving system concats the entire transcript as a long sequence of tokens and runs the entire thing thru forward pass

how this works
- context window = hard limit
    - models can only handle inputs up to a fixed size 
    - if my convo exceeds that, the oldest content has to be dropped or summarized
- token-by-token generation
    - the forward pass doesn't produce a whole response in 1 shot
    - it produces 1 token at a time
- key-value caching is an optimization
    - key value cache for tokens already processed 
    - each new token only requires computing the new parts
- cost scales with context length
    - long convos get expensive
    - stuff the whole code base in the prompt = impossible

LLM = function - a very large math function from "sequence of tokens in" to "next token out"
- convos = interfact trick where we keep feeding the function its own outputs concat with new user inputs
- produces responses that look like memory and reasoning 

transformer architecture added a critical ingredient on top of standard neural networks = attn mechanism 
- let's each token in the input look at every other token and weight how relevant they are to each other
- this is allows "it" in a sentence to know it refers back to a noun mentioned a few words earlier
- attn itself is also implemented using the same fundamentals - matmul, weights, biases, activations

## c1w4 review
initialize paramaters
- weights - set as small random values
- bias - set as zero
for each layer in the neural network, it gets its own weight matrix (aka table)
- rows = how many nodes this layer has = how many outputs it produces
- columns = how many nodes the previous layer has = how many inputs it receives from the previous layer
for each layer in the neural network, it gets its own bias vector (aka list)
- set to zero 

within a single forward pass, weights are fixed
- for each layer l, you take activations (aka inputs) from the previous layer and the current weights and compute
    - z for the layer = weight and input matrix for the layer + bias for the layer
    - a for the layer output = activation function(z)
- weights are not adjusted during forward pass 
- they are used to calculate

across training iterations
1. forward pass with curent weights and bias parameters -> predictions
2. compute the loss = how good are the model predictions?
3. backward pass = determine gradient -> how to adjust parameters to potentially improve prediction
4. update parameters and start another iteration

learning rate hyperparameter determines how to adjust parameters from one iteration to the next
- too small -> training is slow 
- too large -> may overshoot the global minimum

```python
# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    #(≈ 1 line of code)
    ### START CODE HERE ###
    # np.dot is matrix multiplication
    # computes all the Z for all the units across the training examples in single call, with no loops = vectorization
    # we calculate 
    Z = np.dot(W, A) + b
    
    ### END CODE HERE ###
    cache = (A, W, b)
    
    return Z, cache
```

when we iterate, we calculate
- one pre-activation Z per node and example  
- one post-activation A per node and example 

linear backward
- dW - avg per example 

## c1w4 assignment 2
train_x_org.shape (209, 64, 64, 3)
- 209 = number of training images 
- 64, 64 = 64 pixels wide x 64 pixels tall
- 3 = each pixel has 3 color channels

train_y.shape (1, 209)
- 1 label per image - 0 for not-a-cat, 1 for cat
- 209 = one label for each of the 209 training images 

# Course 2
applied ML is a highly iterative process   
idea -> code -> experiment  
hyperparameters are very specific for different approaches/domains   

## Data - train/holdout
setting up train/dev/test sets well can speed up the iterative experiment cycle   

for all data, split up 
- training
    - what the model uses as input for training
    - model sees features and labels together 
    - it adjusts weights to fits the ground truth 
- holdout
    - model only sees the features when predicting
    - then compare its predicts against the held-back labels to measure how well it generalizes
    - used to pick hyperparameters/ compare algo choices
    - needs to be just big enough to evaluate different algo choices
    - k-fold cross-validation can help avoid one fixed holdout by rotating which slice serves as the holdout acrsoss k runs and averages the score 
- test
    - final validation uses data never seen before during the iterative dev cycle
    - this is because holdout data indirectly leaks info into decision during each iterative adjustment that uses holdout for eval
    - provides honest estimate of real world performance

modern big data era 
- 60/20/20 split made sense with smaller datasets
- may be overkill for larger datasets so may make more sense 98% train, 1% holdout, 1% test

something to avoid = mismatched train/test distribution
- ex: avoid having high quality images in the training set only and poor quality images in the dev/test sets only
- want to mix things up 

## Bias/variance in evaluting performance
- high bias = training is underfitting
    - performs poorly on both train and holdout
- high variance = training is overfitting
    - performs well on train set
    - performs poorly on holdout set
- high bias + high variance = worst of both worlds
    - high bias = should have been curved, but was linear line which is underfitting
    - high varaince = overfit on random outliers

## ML basic recipe
1. High bias - performing poorly on training data that includes both features and labels
- Bigger network
    - Too small a network = can't grasp complex patterns
    - More layers/neurons -> can grasp richer, more nonlinear relationship
- Train longer
    - Network has enough capacity, but aren't given enough time to find good enough values for its weights 
    - Just like a person who is more than capable, but just hasn't been given enough time to learn the material
- May need to use a different algo/neural network architecture
2. High variance - got past high bias so training model output performing well, but doesn't perform well on holdout data = overfit
- Get more data (if possible)
- Regularization
- May need a more appropriate neural network architecture
3. Done = no longer have high bias and high variance

