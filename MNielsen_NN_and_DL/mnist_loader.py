#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:30:16 2019

@author: moshiur
source: https://github.com/mnielsen/neural-networks-and-deep-learning

a library to load MNIST image data availale at 
https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/data
.
"""

# import relevant libraries
#import pickle # a data serialization package written in C
import _pickle as cPickle
import gzip # to handle zip files
import numpy as np

def load_data():
    """ return MNIST data as tuples of training data, validation data and test data """
    file = gzip.open('data/mnist.pkl.gz', 'rb')
    #training_data, validation_data, test_data = pickle.load(file)
    """ to resolve the encoding issue in pickle and numpy for Python 3.x use encoding = 'latin1'
    after the file object in pickle/cPickle. For more details see https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3

    """
    training_data, validation_data, test_data = cPickle.load(file, encoding='latin1')
    file.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """returns a tuple of (training_data, validation_data, test_data) with some modificaitons
    that is helpful for using the data in the neural network building.
    
    training_data: 50000 2-tuple of (x1, y1) where
        x1: 784-dimensional numpy.ndarray of input images
        y1: 10-dimansional label of the digit for x1, it can be thought of as a one-hot-encoded
        value of training labels
        
    validation_data: 10000 2-tuple of (x2, y2) where
        x2: 784-dimensional numpy.ndarray of input images
        y2: the digit value (integer) of the corresponding image in x2
    
    test_data: 10000 2-tuple of (x3, y3) where
        x3: 784-dimensional numpy.ndarray of input images
        y3: the digit value (integer) of the corresponding image in x3
        
    """ 
    tr_data, va_data, te_data = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in tr_data[0]]
    training_results = [vectorized_data(y) for y in tr_data[1]]
    training_data = zip(training_inputs, training_results)
    
    valdiation_inputs = [np.reshape(x, (784,1)) for x in va_data[0]]
    validation_data = zip(valdiation_inputs, va_data[1])
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_data[0]]
    test_data = zip(test_inputs, te_data[1])
    
    return (training_data, validation_data, test_data)

def vectorized_data(j):
    """ return a 10-dimensional vector with 1 in the j-th position """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e