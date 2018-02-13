import os
import numpy as np
from . import utilities

"""
Main file for deep neural network classification task.
Holds all the required functions for the classification task.
Some functions in utilities.py as well
"""


def initialize_parameters(layers_dims):
    """
    Initializes the parameters for each layer in Neural Network. Uses the number of nodes in each layer,
    given in  :param layers_dims (layers dimensions), for initialization task.
    The initialization used for weights is He initialization.
    The first element in :param layers_dims is the input size
    :param layers_dims: list holding the size of units in each layer of nn where the first element is size of inputs
    :return: parameter dictionary containing W1, b1, W2, b2, .... WL, bL where L is len of :param layers_dims
             For some l (small L)<=L:
             Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
             bl -- bias vector of shape (layer_dims[l], 1)
    """

    L = len(layers_dims)  # Len of layer_dims

    parameters = {}  # empty dictionary

    # As first element in :param layers_dims is the input size, therefore loop starts from 1 to L
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.random.randn(layers_dims[l], 1)

    return parameters


def linear_forward(A, W, b):
    """
    Computes forward linear pass using input(if first layer)/activation from previous layer
    :param A: The input/activation from previous layer
    :param W: Weight matrix : numpy 2d-array of size: (size of current layer, size of previous layer)
    :param b: bias, numpy vector of size: (size of current layer, 1)
    :return:   Z -- the input for the activation function, also called pre-activation parameter
               Z = W*A + b
               In terms of layer L : Z[L]=W[L]A[L−1]+b[L]
    """

    Z = np.dot(W, A) + b
    return Z


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements the forward pass including the activation function ie. Linear -> Activation
    :param A_prev: The input/activation from previous layer
    :param W: Weight matrix : numpy 2d-array of size: (size of current layer, size of previous layer)
    :param b: bias, numpy vector of size: (size of current layer, 1)
    :param activation: The type of activation to be used for current layer. Either Sigmoid ro RELU
    :return: A - the post-activation value, calculated as Linear -> Activation
             cache - the cache dictionary for A-prev, W, b and Z
    """

    Z = linear_forward(A_prev, W, b)  # The forward linear value Z
    A = None

    if activation == 'sigmoid':
        A = utilities.sigmoid(Z)
    elif activation == 'relu':
        A = utilities.relu(Z)

    cache = {'A_prev_cache': A_prev, 'W_cache': W, 'b_cache': b, 'Z_cache': Z}

    return A, cache


"""
Linear -> Activation computation is considered as one computation/unit in Neural network

For L layered neural network, Linear -> Activation is performed L times
Where structure in our case is: (Linear -> Relu) * L-1 times -> Linear -> Sigmoid
"""


def model_forward(X, parameters):
    """
    Implements the forward propagation which is (Linear -> RELU) * L - 1 times -> Linear -> Sigmoid
    :param X: input data, size: (input size/number of features, number of examples)
    :param parameters: dictionary of initialized parameters
    :return: A_last - the last post-activation value
             caches - a list of caches from each layer
    """

    A = X  # the first input for Neural Network
    caches = []  # initialize an empty list to hold cache coming for each layer computation

    L = len(parameters) // 2  # To get number of layers in NN, each layer has 2 parameters W and b

    # Implement [LINEAR -> RELU]*(L-1).
    # Add "cache" to the "caches" list. Caches will be used during back propagation
    for layer in range(1, L - 1):
        W = parameters['W' + str(layer)]  # W for layer 'layer'
        b = parameters['b' + str(layer)]  # b for layer 'layer'
        A, cache = linear_activation_forward(A_prev=A, W=W, b=b, activation='relu')
        caches.append(cache)

    # Implement LINEAR -> SIGMOID.
    # Add "cache" to the "caches" list.
    # L corresponds to last layer. Last computed A in for loop above is used for Linear -> Sigmoid
    A_last, cache = linear_activation_forward(A_prev=A, W=parameters['W' + str(L)], b=parameters['b' + str(L)],
                                              activation='sigmoid')
    caches.append(cache)

    return A_last, caches
