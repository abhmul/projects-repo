import tensorflow as tf
from functools import partial

from ..common import assertions, functional as F


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = F.heaveside(strength)
        return activation, strength


class SignLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = F.sign(strength)
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

    def update(self, input_tensor, label_tensor, learning_rate=1.0):
        output = self(input_tensor)  # B x 1
        prediction = output["activations"][-1]  # B x 1
        error = self.error_vector(prediction, label_tensor)

        # Get the internal labels
        labels = [label_tensor]
        for layer in [self.final_layer] + self.hidden_layers:
            F.sign(layer.kernel)

