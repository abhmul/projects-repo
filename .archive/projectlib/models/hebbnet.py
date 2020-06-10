import tensorflow as tf
from functools import partial

from ..common import assertions, functional


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = functional.heaveside(strength)
        return activation, strength


def base_loss_simple(strength, label):
    batch_loss = tf.keras.backend.batch_dot(-label, strength)
    assert batch_loss.shape[0] == label.shape[0]
    assert batch_loss.shape[1] == 1

    return tf.reduce_mean(batch_loss)


def base_loss_max_margin(strength, label):
    batch_loss = tf.maximum(0, 1 - tf.keras.backend.batch_dot(label, strength))
    assert batch_loss.shape[0] == label.shape[0]
    assert batch_loss.shape[1] == 1

    return tf.reduce_mean(batch_loss)


# Base loss function is -yTWx = -yTs
# Weight is inserted in between y and s
def loss_a(
    strength, label, base_loss=base_loss_simple, weight=None, regularization=0.0
):
    """The scaled linear loss function, gradients are kept through weights"""
    assertions.assert_is_signs(label)
    assert label.ndim == 2
    assert strength.ndim == 2
    # label is B x n and each elem in {-1, 1}
    # strength is B x m (if no weight, m = n)
    # weight is n x m
    if weight is not None:
        assert tuple(weight.shape) == (strength.shape[1], label.shape[1])
        strength = tf.matmul(strength, weight)
    # strength should now be B x n
    return base_loss(strength, label) + regularization


def loss_b(
    strength, label, base_loss=base_loss_simple, weight=None, regularization=0.0
):
    weight = None if weight is None else tf.stop_gradient(weight)
    return loss_a(
        strength,
        label,
        base_loss=base_loss,
        weight=weight,
        regularization=regularization,
    )


def loss_c(
    strength, label, base_loss=base_loss_simple, weight=None, regularization=0.0
):
    weight = None if weight is None else tf.stop_gradient(tf.sign(weight))
    return loss_a(
        strength,
        label,
        base_loss=base_loss,
        weight=weight,
        regularization=regularization,
    )


def create_loss_fn(loss_fn, base_loss):
    print(f"Using base loss: {base_loss.__name__}")
    return partial(loss_fn, base_loss=base_loss)


class HebbNetSimple(tf.keras.Model):
    def __init__(self, loss_func, regularization=1.0):
        super(HebbNetSimple, self).__init__()
        print("This HebbNet has only 1 output!")
        self.regularizer = tf.keras.regularizers.l2(regularization)
        self.layer = HeavesideLayer(1, kernel_regularizer=self.regularizer)

        self.loss_func = loss_func

    def call(self, input_tensor, training=False):
        activation, strength = self.layer(input_tensor)
        return {"activations": [activation], "strengths": [strength]}

    def pred(self, input_tensor):
        return self(input_tensor)["activations"][-1]

    def loss(self, input_tensor, label):
        output = self(input_tensor)
        strength = output["strengths"][-1]

        loss_score = self.loss_func(strength, label, regularization=self.layer.losses)
        print(f"loss score: {loss_score}")
        return loss_score


class HebbNet2Layer(tf.keras.Model):
    def __init__(self, n_hidden, loss_func, regularization=1.0):
        super(HebbNet2Layer, self).__init__()
        print("This HebbNet has only 1 output!")
        self.regularizer = tf.keras.regularizers.l2(regularization)
        self.first_layer = HeavesideLayer(n_hidden, kernel_regularizer=self.regularizer)
        self.second_layer = HeavesideLayer(1, kernel_regularizer=self.regularizer)
        print("Not building layers!")

        self.loss_func = loss_func

    def call(self, input_tensor, training=False):
        x1, s1 = self.first_layer(input_tensor)
        x2, s2 = self.second_layer(x1)

        return {"activations": [x1, x2], "strengths": [s1, s2]}

    def pred(self, input_tensor):
        return self(input_tensor)["activations"][-1]

    def loss(self, input_tensor, label):
        output = self(input_tensor)

        s1 = output["strengths"][-1]
        loss_score1 = self.loss_func(s1, label, regularization=self.second_layer.losses)

        s2 = output["strengths"][-2]
        loss_score2 = self.loss_func(
            s2,
            label,
            weight=self.second_layer.kernel,
            regularization=self.first_layer.losses,
        )

        loss_score = loss_score1 + loss_score2
        print(f"loss score: {loss_score}")
        return loss_score
