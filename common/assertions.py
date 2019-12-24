import tensorflow as tf


def assert_is_signs(tensor):
    tf.Assert(tf.reduce_all((tensor == 1) | (tensor == -1)), [tensor])


def assert_is_signs_or_zero(tensor):
    tf.Assert(tf.reduce_all((tensor == 1) | (tensor == -1) | (tensor == 0)), [tensor])


def assert_is_binary(tensor):
    tf.Assert(tf.reduce_all((tensor == 0) | (tensor == 1)), [tensor])
