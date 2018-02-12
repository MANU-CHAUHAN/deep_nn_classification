import os
import h5py
import numpy as np


def load_cat_dataset():
    """
    Load cat vs not-cat data set and return train and test sets and list of classes
    :return: train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    """
    train_dataset = h5py.File('cat_not_cat_dataset/train_catvsnotcat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    test_dataset = h5py.File('cat_not_cat_dataset/test_catvsnotcat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):
    """
    Calculates the Sigmoid activation function for the given parameter
    :param z: Output of the linear layer, of any shape
    :return: Post-activation value, the output of sigmoid function on :param z
    """
    return 1 / (1 + np.exp(-z))


def relu(z):
    """
    Calculates the RELU activation function for the given parameter
    :param z: Output of the linear layer, of any shape
    :return: Post-activation value, the output of RELU activation on :param z
    """
    return np.maximum(0, z)


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Calculates the gradient of cost wrt. Z (Z is the pre-activation linear value)
    :param dA: The Post-activation gradient
    :param cache: the cache value storing 'Z', used for calculating backward propagation
    :return: dZ - the gradient of cost wrt. Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Calculates the gradient of cost wrt. Z (Z is the pre-activation linear value)
    :param dA: The Post-activation gradient
    :param cache: the cache value storing 'Z', used for calculating backward propagation
    :return: dZ - the gradient of cost wrt. Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)

    # When the value, Z<= 0 , the gradient is = 0
    dZ[Z <= 0] = 0
    return dZ
