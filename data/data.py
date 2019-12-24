"""
Loads datasets as tf.data.Dataset
"""
import logging
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as keras

from ..common import plotting


def dataset_to_numpy(dataset):
    length = dataset.reduce(0, lambda x, _: x + 1).numpy()
    dataset = dataset.batch(length)
    for X, y in dataset:
        return X.numpy(), y.numpy()


def standardize_data(X):
    return StandardScaler().fit_transform(X)


def generate_blobs():
    X, y = datasets.make_blobs(centers=2)
    X = standardize_data(X)
    logging.info(
        f"Generated blobs with {X.shape[0]} samples : {X.shape[1]} features : {len(np.unique(y))} labels"
    )
    train = tf.data.Dataset.from_tensor_slices((X, y))
    return {"train": train}


def generate_moons():
    X, y = datasets.make_moons(200, noise=0.2)
    X = standardize_data(X)
    logging.info(
        f"Generated blobs with {X.shape[0]} samples : {X.shape[1]} features : {len(np.unique(y))} labels"
    )
    train = tf.data.Dataset.from_tensor_slices((X, y))
    return {"train": train}


def generate_circles():
    X, y = datasets.make_circles(200, noise=0.2)
    X = standardize_data(X)
    logging.info(
        f"Generated blobs with {X.shape[0]} samples : {X.shape[1]} features : {len(np.unique(y))} labels"
    )
    train = tf.data.Dataset.from_tensor_slices((X, y))
    return {"train": train}


def generate_xor():
    X = np.asarray([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    y = np.asarray([0, 1, 1, 0])
    train = tf.data.Dataset.from_tensor_slices((X, y))
    return {"train": train}


def mnist():
    train, test = keras.datasets.mnist.load_data()
    train = tf.data.Dataset.from_tensor_slices(train)
    test = tf.data.Dataset.from_tensor_slices(test)
    return {"train": train, "test": test}
