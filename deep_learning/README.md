# [Deep Learning Specialization](https://learn.deeplearning.ai/specializations/deep-learning/information#course-outline)

## What is a neural network?
Non-linear function
- graph isn't a straight line
- the relationship changes at different points
- ReLU = rectified linear unit
    - negatives -> 0
    - positives -> unchanges 
    - avoids negative numbers
    - ex: housing price prediction doesn't go below zero

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