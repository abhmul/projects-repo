import numpy as np
import tensorflow as tf


def mix_classification_datasets(*datasets):
    input_datasets, target_datasets = zip(*datasets)
    return np.concatenate(list(input_datasets)), np.concatenate(list(target_datasets))


def sample(dataset, m, rng, replace=True):
    if replace:
        return sample_with_replacement(dataset, m, rng)
    else:
        return sample_without_replacement(dataset, m, rng)

def sample_with_replacement(dataset, m, rng):
    X, Y = dataset
    inds = rng.integers(len(Y), size=m)
    return X[inds], Y[inds], inds

def sample_without_replacement(dataset, m, rng):
    X, Y = dataset
    inds = rng.permutation(len(Y))[:m]
    return X[inds], Y[inds], inds


def retrieve_split(dataset, inds):
    X, Y = dataset
    bool_arr = np.zeros(len(Y), dtype=bool)
    bool_arr[inds] = 1
    return (X[bool_arr], Y[bool_arr]), (X[~bool_arr], Y[~bool_arr])


def flatten_data(X):
    return X.reshape(len(X), -1)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    return (x_train, y_train), (x_test, y_test)
