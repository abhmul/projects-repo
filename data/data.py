"""
Loads datasets as tf.data.Dataset
"""
import logging
import numpy as np
from sklearn import datasets
import tensorflow as tf
import tensorflow.keras as keras


def generate_blobs():
    X, y = datasets.make_blobs(centers=2)
    logging.info(
        f"Generated blobs with {X.shape[0]} samples : {X.shape[1]} features : {len(np.unique(y))} labels"
    )
    train = tf.data.Dataset.from_tensor_slices((X, y))
    return {"train": train}


def mnist():
    train, test = keras.datasets.mnist.load_data()
    train = tf.data.Dataset.from_tensor_slices(train)
    test = tf.data.Dataset.from_tensor_slices(test)
    return {"train": train, "test": test}
