import tensorflow as tf

from ..common import functional as F


def minimize_binary_label(inputs, labels, model, optimizer, num_iter=100):
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels = F.binary_to_signed(tf.convert_to_tensor(labels, dtype=tf.float32))
    if labels.ndim == 1:
        labels = tf.expand_dims(labels, -1)

    loss_fn = lambda: model.loss(inputs, labels)  # noqa: E731
    var_list_fn = lambda: model.trainable_weights  # noqa: E731

    for _ in range(num_iter):
        optimizer.minimize(loss_fn, var_list_fn)
