import tensorflow as tf
from ..common import assertions, functional as F


@tf.custom_gradient
def heaveside_linear_ste(x):
    def grad(dy):
        return dy

    return F.heaveside(x), grad


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = heaveside_linear_ste(strength)
        return activation, strength


@tf.custom_gradient
def sign_linear_ste(x):
    def grad(dy):
        return dy

    return F.sign(x), grad


class SignLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = sign_linear_ste(strength)
        return activation, strength