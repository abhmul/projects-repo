import numpy as np
import tensorflow as tf
import tf_assert as assertions


def binary_to_signed(binary_tensor):
    assertions.assert_is_binary(binary_tensor)
    return 2 * binary_tensor - 1


def signed_to_binary(signed_tensor):
    assertions.assert_is_signs(signed_tensor)
    return (signed_tensor + 1) / 2


def heaveside(x: tf.Tensor):
    return tf.stop_gradient(tf.cast(tf.greater_equal(x, 0), tf.float32))


def sign(x: tf.Tensor):
    return tf.stop_gradient(2 * heaveside(x) - 1)


def np_unravel(flat_idxs, shape):
    return np.stack(np.unravel_index(flat_idxs, shape), axis=-1)