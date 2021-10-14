import numpy as np
import tensorflow as tf


def mix_datasets(train, test):
    x_train, y_train = train
    x_test, y_test = test
    return np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])


def sample_train(dataset, m, rng):
    X, Y = dataset
    inds = rng.integers(len(Y), size=m)
    return X[inds], Y[inds]


def flatten_data(X):
    return X.reshape(len(X), -1)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    return (x_train, y_train), (x_test, y_test)


def mnist_mlp_connector(dataset):
    (x_train, y_train), (x_test, y_test) = dataset
    x_train = flatten_data(x_train)
    x_test = flatten_data(x_test)
    assert x_train.shape == (60000, 28*28)
    assert x_test.shape == (10000, 28*28)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    return (x_train, y_train), (x_test, y_test)
