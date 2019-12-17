"""
Loads datasets as tf.data.Dataset
"""
import tensorflow as tf
import tensorflow.keras as keras


def mnist():
    train, test = keras.datasets.mnist.load_data()
    train = tf.data.Dataset.from_tensor_slices(train)
    test = tf.data.Dataset.from_tensor_slices(test)
    return train, test
