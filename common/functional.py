import tensorflow as tf


def binary_to_signed(binary_tensor):
    return 2 * binary_tensor - 1


def heaveside(x: tf.Tensor):
    return tf.stop_gradient(tf.cast(tf.greater_equal(x, 0), tf.float32))
