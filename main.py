import os
import pickle
import numpy as np
from . import utilities
from matplotlib import pyplot as plt

"""
Main file for deep neural network classification task.
Holds all the required functions for the classification task.
Some functions in utilities.py as well
"""

hidden_layers = [200, 100, 50]  # hidden layers only, ie without input and output layer sizes


def initialize_parameters(layers_dims):
    """
    Initializes the parameters for each layer in Neural Network. Uses the number of nodes in each layer,
    given in  :param layers_dims (layers dimensions), for initialization task.
    The initialization used for weights is He initialization.
    The first element in :param layers_dims is the input size
    :param layers_dims: list holding the size of units in each layer of nn where the first element is size of inputs
    :return: parameter dictionary containing W1, b1, W2, b2, .... WL, bL
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

    # Implement [LINEAR -> RELU] * (L-1) times.
    # Add "cache" to the "caches" list. Caches will be used during back propagation
    for layer in range(1, L):
        W = parameters['W' + str(layer)]  # W for current 'layer'
        b = parameters['b' + str(layer)]  # b for current 'layer'
        A, cache = linear_activation_forward(A_prev=A, W=W, b=b, activation='relu')
        caches.append(cache)

    # Implement LINEAR -> SIGMOID.
    # Add "cache" to the "caches" list.
    # L corresponds to last layer. Last computed A in above for-loop is used for Linear -> Sigmoid
    A_last, cache = linear_activation_forward(A_prev=A, W=parameters['W' + str(L)], b=parameters['b' + str(L)],
                                              activation='sigmoid')
    caches.append(cache)

    return A_last, caches


def compute_cost(A_last, Y, parameters, lambd):
    """
    Calculates the cost for the network. The optimization process tries to reduce the loss for the network.
    Uses L-2 regularization where lambda for regularization is given by :param lambd
    :param A_last: The post-activation value from the last layer, the output of the network
    :param Y: The true labels for all the samples
    :param parameters: The parameter dictionary, used for L2 regularization
    :param lambd: The value of lambda in L2 regularization
    :return: The cost for the network, type float
    """
    m = Y.shape[1]

    # use axis=1 for sum across rows and keepdims help to get shape in proper format such as (2,1) instead of (2,)
    cost = -1 / m * np.sum((np.multiply(Y, np.log(A_last)) + np.multiply((1 - Y), np.log(1 - A_last))), axis=1,
                           keepdims=True)

    # np.squeeze() is used to convert a value such as [[100]] to 100
    cost = np.squeeze(cost)

    # Use regularized cost if lambda for L2 is > 0.0
    if lambd > 0.0:
        L = len(parameters) // 2
        regularization_cost = (lambd / 2 * m) * np.sum(
            np.sum(np.square(parameters['W' + str(l)]) for l in range(1, L + 1)))
        cost = cost + regularization_cost
    return cost


def linear_backward(dZ, cache, lambd):
    """
    Implement linear part for backward propagation for a single layer
    :param dZ: The gradient of cost wrt. Z for current layer
    :param cache: dictionary containing 'A_prev_cache', 'W_cache', 'b_cache' and 'Z_cache'
                  for current layer, computed during forward propagation
    :param lambd: the L2 regularization lambda value
    :return: dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
             dW -- Gradient of the cost with respect to W (for current layer l), same shape as W
             db -- Gradient of the cost with respect to b (for current layer l), same shape as b
    """

    A_prev, W, b, Z = cache['A_prev_cache'], cache['W_cache'], cache['b_cache'], cache['Z_cache']

    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    if lambd > 0.0:
        dW = dW + (lambd / m) * W

    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    # assert to cross check dimensions
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    :param dA: post-activation gradient for current layer
    :param cache: cache for computing backward propagation
    :param activation: type of activation to be used, Sigmoid or RELU
    :param: lambda used for L2 regularization
    :return:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    Z = cache['Z_cache']
    dZ = None

    if activation == 'sigmoid':
        dZ = utilities.sigmoid_backward(dA=dA, cache=Z)
    elif activation == 'relu':
        dZ = utilities.relu_backward(dA=dA, cache=Z)

    dA_prev, dW, db = linear_backward(dZ, cache, lambd)

    return dA_prev, dW, db


def model_backward(AL, Y, caches, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) times -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if false else 1 if true)
    lambd -- the regularization parameter, range: 0.0 - 1.0
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
    """
    grads = {}

    Y = Y.reshape(AL.shape)

    L = len(caches)

    # last sigmoid layer dAL is given by -Y/A - (1 - Y)/(1 - A)
    dAL = - np.divide(Y, AL) - np.divide((1 - Y), (1 - AL))

    last_layer_cache = caches[L - 1]

    # last layer grads
    dA_prev, dW, db = linear_activation_backward(dA=dAL, cache=last_layer_cache, activation='sigmoid', lambd=lambd)
    grads['dA' + str(L)] = dA_prev
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db

    # grads for other L - 1 layers
    for layer in reversed(range(1, L)):
        current_layer_cache = caches[layer - 1]  # caches is a list -> 0 indexed

        dA_prev, dW, db = linear_activation_backward(dA=grads['dA' + str(layer + 1)], cache=current_layer_cache,
                                                     activation='relu', lambd=lambd)
        grads['dA' + str(layer)] = dA_prev
        grads['dW' + str(layer)] = dW
        grads['db' + str(layer)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates the parameters using the calculated grads and given learning rate
    :param parameters: the dictionary of all parameters
    :param grads: the dictionary of grads calculated during back propagation
    :param learning_rate: learning rate for gradient descent
    :return: updated dictionary of parameters
    """

    L = len(parameters) // 2  # number of layers

    for layer in range(1, L + 1):
        parameters['W' + str(layer)] = parameters['W' + str(layer)] - learning_rate * grads['dW' + str(layer)]
        parameters['b' + str(layer)] = parameters['b' + str(layer)] - learning_rate * grads['db' + str(layer)]

    return parameters


def neural_network_model(X, Y, layers_dims, iterations=1000, learning_rate=0.001, lambd=0.0):
    """
    Complete model connecting other parts
    :param X: Input data
    :param Y: Truth labels
    :param layers_dims: the list containing the layer sizes including the input layer (1st element in layers_dims)
    :param iterations: Number of iterations to run for gradient descent, default 1000
    :param learning_rate: the learning rate to be used during gradient descent, default 0.001
    :return: dictionary of updated parameters
    """
    parameters = initialize_parameters(layers_dims=layers_dims)

    costs = []
    iteration_numbers = []

    for i in range(iterations):
        A_last, caches = model_forward(X=X, parameters=parameters)

        if i % 100 == 0:
            cost = compute_cost(A_last=A_last, Y=Y, parameters=parameters, lambd=lambd)
            costs.append(cost)
            iteration_numbers.append(i)
            print('Cost after %d iterations is %f' % (i, cost))

        grads = model_backward(AL=A_last, Y=Y, caches=caches, lambd=lambd)

        parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)

    return parameters, iteration_numbers, costs


def predict(X, parameters):
    """
    Predict on data provided using trained parameters
    :param X: The data for which to predict the labels
    :param parameters: learned parameters
    :return: array of predictions
    """

    A_last, caches = model_forward(X, parameters)
    result = np.zeros(shape=(1, X.shape[1]))
    for i in range(A_last.shape[1]):
        result[0][i] = 1 if A_last[0][i] > 0.5 else 0
    return result


def run_nn_model(args):
    """ Run the neural network model after collecting the data and user given inputs """

    if len(args) == 1 or args[1] not in ['train', 'test', 'predict']:
        raise Exception('\n Missing argument or Wrong argument passed')
    else:
        if args[1].lower() == 'train':
            train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = utilities.load_cat_dataset()

            # reshape train/test data set from (sample_numbers, height, width, 3) to (height*width*3, sample_numbers)
            #  3 dimension corresponds to RGB color channel values
            train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
            train_x = train_set_x_flatten / 255  # normalize inputs: dividing by max value for a pixel

            layers_dimensions = [train_x.shape[0]] + globals()['hidden_layers'] + [1]

            trained_parameters, iterations, costs = neural_network_model(X=train_x, Y=train_set_y_orig,
                                                                         layers_dims=layers_dimensions, iterations=100,
                                                                         learning_rate=0.001, lambd=0.7)
            with open('trained_parameters.pkl', 'wb') as file:
                pickle.dump(trained_parameters, file)

            plt.plot(iterations, costs)
            plt.xlabel('Number of Iterations')
            plt.ylabel('Training Cost')
            plt.show()

        elif args[1].lower() == 'test':
            if not os.path.isfile('trained_parameters.pkl'):
                raise Exception('\n Trained parameter pickle file is not present.')

            train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = utilities.load_cat_dataset()

            # reshape train/test data set from (sample_numbers, height, width, 3) to (height*width*3, sample_numbers)
            #  3 dimension corresponds to RGB color channel values
            test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
            test_set_x = test_set_x_flatten / 255  # normalize inputs: dividing by max value for a pixel

            with open('trained_parameters.pkl', 'rb') as file:
                trained_parameters = pickle.load(file)

            predictions = predict(X=test_set_x, parameters=trained_parameters)

            print('\n \n Accuracy for test set data: %f' % np.mean(predictions == test_set_y_orig) * 100)
            
