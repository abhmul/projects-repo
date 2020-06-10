import tensorflow as tf
from functools import partial

from ..common import assertions, functional as F

nll = tf.nn.sigmoid_cross_entropy_with_logits


def loss(label, strength, weight=None):
    assertions.assert_is_binary(label)
    assert label.ndim > 1
    batch_size = label.shape[0]
    loss_val = nll(label, strength)  # B x n
    if weight is not None:
        loss_val = tf.stop_gradient(weight) * loss_val
    return tf.reduce_sum(loss_val) / batch_size


class HeavesideLayer(tf.keras.layers.Dense):
    def build(self, input_shape):
        super().build(input_shape)
        # self.kernel.assign(F.sign(tf.random_uniform_initializer()(self.kernel.shape)))
        # self.kernel.assign(
        #     F.sign(tf.random_uniform_initializer()(self.kernel.shape)) / 1000
        # )
        # self.kernel.assign(tf.zeros_initializer()(self.kernel.shape))
        self.bias.assign(tf.zeros_initializer()(self.bias.shape))

    def call(self, x):
        strength = super().call(x)
        activation = F.heaveside(strength)
        return activation, strength


class SigmoidLayer(tf.keras.layers.Dense):
    def build(self, input_shape):
        super().build(input_shape)
        # self.kernel.assign(
        #     F.sign(tf.random_uniform_initializer()(self.kernel.shape)) / 1000
        # )
        # self.kernel.assign(tf.zeros_initializer()(self.kernel.shape))
        self.bias.assign(tf.zeros_initializer()(self.bias.shape))

    def call(self, x):
        strength = super().call(x)
        activation = tf.stop_gradient(tf.nn.sigmoid(strength))
        return activation, strength


class BackpropSigmoidLayer(tf.keras.layers.Dense):
    def build(self, input_shape):
        super().build(input_shape)
        # self.kernel.assign(
        #     F.sign(tf.random_uniform_initializer()(self.kernel.shape)) / 1000
        # )
        # self.kernel.assign(tf.zeros_initializer()(self.kernel.shape))
        self.bias.assign(tf.zeros_initializer()(self.bias.shape))

    def call(self, x):
        strength = super().call(x)
        activation = tf.nn.sigmoid(strength)
        return activation, strength


class BackpropNet1Layer(tf.keras.Model):
    def __init__(self, n_hidden, loss_func=loss):
        super(BackpropNet1Layer, self).__init__()
        print("This SigmoidNet has only 1 output!")
        print("This SigmoidNet has only 1 hidden layer!")
        self.hidden_layer = BackpropSigmoidLayer(n_hidden)
        self.final_layer = BackpropSigmoidLayer(1)

        self.loss_func = loss_func

    def call(self, input_tensor, training=False):
        activations = []
        strengths = []
        x = input_tensor
        for layer in [self.hidden_layer] + [self.final_layer]:
            x, s = layer(x)
            activations.append(x)
            strengths.append(s)

        return {"activations": activations, "strengths": strengths}

    def pred(self, input_tensor):
        activations = []
        strengths = []
        x = input_tensor
        for layer in [self.hidden_layer] + [self.final_layer]:
            x, s = layer(x)
            x = tf.cast(x >= 0.5, dtype=tf.float32)
            activations.append(x)
            strengths.append(s)

        return tf.cast(activations[-1] >= 0.5, dtype=tf.float32)

    def loss(self, label_tensor, input_tensor):
        batch_size = label_tensor.shape[0]
        outputs = self(input_tensor)
        strength = outputs["strengths"][-1]
        return tf.reduce_sum(nll(label_tensor, strength)) / batch_size


class SigmoidNet1Layer(tf.keras.Model):
    def __init__(self, n_hidden, loss_func=loss):
        super(SigmoidNet1Layer, self).__init__()
        print("This SigmoidNet has only 1 output!")
        print("This SigmoidNet has only 1 hidden layer!")
        self.hidden_layer = SigmoidLayer(n_hidden)
        self.final_layer = SigmoidLayer(1)
        # self.hidden_layer = HeavesideLayer(n_hidden)
        # self.final_layer = HeavesideLayer(1)

        self.loss_func = loss_func

    def call(self, input_tensor, training=False):
        activations = []
        strengths = []
        x = input_tensor
        for layer in [self.hidden_layer] + [self.final_layer]:
            x, s = layer(x)
            activations.append(x)
            strengths.append(s)

        return {"activations": activations, "strengths": strengths}

    def pred(self, input_tensor):
        return tf.cast(self(input_tensor)["activations"][-1] >= 0.5, dtype=tf.float32)

    def loss(self, label_tensor, input_tensor):
        assertions.assert_is_binary(label_tensor)
        output = self(input_tensor)

        final_strength = output["strengths"][-1]
        loss_score1 = self.loss_func(label_tensor, final_strength)

        # get the loss weight
        a1 = output["activations"][-1]
        weight = tf.abs(
            tf.matmul(
                label_tensor * (1 - a1) - (1 - label_tensor) * a1,
                tf.transpose(self.final_layer.kernel),
            )
        )
        # if sign(kernel) == 0, the weight will also be 0, so we have no gradient
        hidden_label = F.signed_to_binary(
            tf.matmul(
                F.binary_to_signed(label_tensor),
                tf.transpose(F.sign(self.final_layer.kernel)),
            )
        )

        hidden_strength = output["strengths"][-2]
        loss_score2 = self.loss_func(hidden_label, hidden_strength, weight=weight)

        loss_score = loss_score1 + loss_score2
        loss_dict = {
            "loss": loss_score,
            "loss_final": loss_score1,
            "loss_hidden": loss_score2,
        }
        # print(loss_dict)

        return loss_dict
