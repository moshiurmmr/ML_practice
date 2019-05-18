#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:27:58 2019

@author: moshiur
source: https://github.com/mnielsen/neural-networks-and-deep-learning
"""

"""
network.py

This is a module from the Neural Networks and Deep Learning book by Michael Nielsen.
It implements the following
- a feedforward neural network
- stochastic gradient descent
- mini-batch update
- backpropagation 
"""

import random
import numpy as np

# network class
class Network():
    def __init__(self, sizes):
        """ The list contains the number of neurons in each layer of the network. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # play with biases to start from i/p layer, i.e., ... fory in sizes[:-1]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        """ Return the output of the network when 'a' is the input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid (np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """ divide the training data into batches of size mini_batch_size and
        update biases and weights using the update_mini_batch function"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                    training_data[k:k + mini_batch_size]
                    for k in range(0, n, mini_batch_size)
                    ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    def update_mini_batch(self, mini_batch, eta):
        """ update network weights and biases by applying gradient descent using backpropgation
        to a single mini batch. Here eta is the learning rate"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # calculate gradients for each sample (x,y) in the mini_batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dwb for nw, dwb in zip(nabla_w, delta_nabla_w)]
        # update weights and biases for the mini batch
        self.weights = [w - (eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
    def backprop(self, x, y,  eta):
        """ returns the tuple (nabla_b, nabla_w) representing the derivate of the cost function. """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward in the network
        activation = x
        # list to store all activation layer-by-layer
        activations = [x]
        # list to store all z (= w.x + b) values layer-by-layer
        zs = [] 
        # calcualte activation of the network
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass in the network
        # error at the output layer transferred to the input of the output
        delta = self.cost_derivate(activations[-1], y) * sigmoid_prime(zs[-1]) # equation-(BP1)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transose())
        # nabla_b and nabla_w calculations from the second to last to first layer
        for l in range(2, self.num_layers):
            z = zs [-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp # equation-(BP2)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l + 1].transpose())
            return(nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """ returns the number of test inputs for wich NN outputs the correct restults """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for x, y in test_data]
        return sum(int(x=y) for (x, y) in test_results)
        
    def cost_derivate(self, output_activation, y):
        """returns the vector out output error """
        return (output_activation - y)
    
# sigmoid activation function
def sigmoid(z):
    """ the sigmoid function """
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """ derivative of sigmoid function """
    return sigmoid(z) * (1 - sigmoid(z))
        