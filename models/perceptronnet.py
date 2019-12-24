import tensorflow as tf
from functools import partial

from ..common import assertions, functional


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = functional.heaveside(strength)
        return activation, strength


class SignLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = functional.sign(strength)
        return activation, strength


class PerceptronNet(tf.keras.Model):
    def __init__(self, n_hiddens, regularization=0.0):
        super(PerceptronNet, self).__init__()
        print("This PerceptronNet has only 1 output!")
        self.regularizer = tf.keras.regularizers.l2(regularization)
        self.hidden_layers = [
            SignLayer(n_hidden, kernel_regularizer=self.regularizer)
            for n_hidden in n_hiddens
        ]
        self.final_layer = SignLayer(1, kernel_regularizer=self.regularizer)
        print("Not building layers!")

    def call(self, input_tensor, training=False):
        activations = []
        strengths = []
        x = input_tensor
        for layer in self.hidden_layers + [self.final_layer]:
            x, s = layer(x)
            activations.append(x)
            strengths.append(x)

        return {"activations": activations, "strengths": strengths}

    def pred(self, input_tensor):
        return self(input_tensor)["activations"][-1]

    @staticmethod
    def error_vector(prediction, label_tensor):
        assertions.assert_is_signs(label_tensor)
        assertions.assert_is_signs(prediction)
        return (label_tensor - prediction) / 2

    def error(self, input_tensor, label_tensor):
        output = self(input_tensor)
        prediction = output["activations"][-1]

