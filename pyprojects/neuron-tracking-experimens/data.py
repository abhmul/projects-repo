import numpy as np
import tensorflow as tf

from projectslib.data import flatten_data

# Just does some assertion checks to make sure everything looks good
def mnist_mlp_connector(dataset):
    (x_train, y_train), (x_test, y_test) = dataset
    x_train = flatten_data(x_train)
    x_test = flatten_data(x_test)
    assert x_train.shape == (60000, 28*28)
    assert x_test.shape == (10000, 28*28)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    return (x_train, y_train), (x_test, y_test)
