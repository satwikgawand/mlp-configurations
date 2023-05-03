# Multi-Layer Perceptron Configurations

---

## Introduction

This repository is a collection of experimental Python notebooks where each one deals with a different hyperparameter and observes the effect different values have on the model's performance. The configuration for every variation is the same except for the parameter being experimented upon.

The experiment utilizes a baseline Sequential Multi-Layer Perceptron. The parameters being tested are:
1. [Activation Functions](#activation-functions)
2. batch size
3. epochs
4. kernel initializers
5. learning rate
6. loss functions
7. network architectures
8. optimizers

---

## File Structure:

- mlp-configurations (repo)
    - parameter
        - parameter.ipynb
        - parameter_result.csv
        - figures
            - all the figures generated from the notebook

---

## Activation Functions

An activation function is a function employed in each neuron inside an ANN which outputs a value for the correspondsing input based on a few different pre-defined methods.

In this experiment, the effect of 9 different activation functions was evaluated:
1. relu
2. sigmoid
3. softmax
4. softplus
5. softsign
6. tanh
7. selu
8. elu
9. exponential

Read more on Keras Documentation: [https://keras.io/api/layers/activations/](https://keras.io/api/layers/activations/)

These activation functions were used on all the layers except for the output layer which exclusively uses the softmax activation function.

 Index | Activation Function | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy | Time (s) 
-------|---------------------|---------------|-------------------|-----------------|---------------------|-----------|---------------|----------
 6     | softmax             | 0.43          | 0.84              | 0.41            | 0.86                | 0.44      | 0.84          | 60.46    
 4     | softsign            | 0.39          | 0.84              | 0.35            | 0.85                | 0.39      | 0.83          | 63.49    
 8     | relu                | 0.45          | 0.83              | 0.42            | 0.84                | 0.46      | 0.82          | 57.05    
 1     | elu                 | 0.7           | 0.5               | 0.7             | 0.49                | 0.7       | 0.5           | 60.42    
 2     | selu                | 0.92          | 0.5               | 0.94            | 0.49                | 0.92      | 0.5           | 62.12    
 3     | tanh                | 0.7           | 0.5               | 0.7             | 0.49                | 0.7       | 0.5           | 55.92    
 5     | softplus            | 0.69          | 0.5               | 0.69            | 0.49                | 0.69      | 0.5           | 65.51    
 0     | exponential         | NaN           | 0.5               | NaN             | 0.51                | NaN       | 0.5           | 55.93    
 7     | sigmoid             | 0.69          | 0.5               | 0.69            | 0.51                | 0.69      | 0.5           | 84.1     
  

We can see from the above table that using __softmax__ as the activation function for the input and hidden layers 