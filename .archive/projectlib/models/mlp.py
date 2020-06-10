import tensorflow as tf
from ..common import assertions, functional as F


class ReluLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = tf.nn.relu(strength)
        return activation, strength


def crossentropy_loss(label_tensor, strengths):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(label_tensor, strengths)
    )


def hinge_loss(label_tensor, strengths):
    return tf.reduce_mean(tf.keras.losses.hinge(label_tensor, strengths))


class MLP(tf.keras.Model):
    def __init__(
        self,
        n_hiddens,
        loss_fn=crossentropy_loss,
        regularization=0.0,
        layer_construtor=ReluLayer,
    ):
        super(MLP, self).__init__()
        print("This MLP has only 1 output!")

        print(f"Using loss fn {loss_fn.__name__}")
        self.loss_fn = loss_fn

        print(f"Using regularization {regularization}")
        self.regularizer = tf.keras.regularizers.l2(regularization)

        print(f"Constructing layers with name {layer_construtor.__name__}")
        print(f"Hideen sizes are {n_hiddens}")

        self.hidden_layers = [
            layer_construtor(n_hidden, kernel_regularizer=self.regularizer)
            for n_hidden in n_hiddens
        ]
        self.final_layer = layer_construtor(1, kernel_regularizer=self.regularizer)
        print("Not building layers!")

    @property
    def all_layers(self):
        return self.hidden_layers + [self.final_layer]

    def call(self, input_tensor, training=False):
        activations = []
        strengths = []
        x = input_tensor
        for layer in self.all_layers:
            x, s = layer(x)
            activations.append(x)
            strengths.append(s)

        return {"activations": activations, "strengths": strengths}

    def pred(self, input_tensor):
        return F.heaveside(self(input_tensor)["strengths"][-1])

    def loss(self, label_tensor, input_tensor):
        output = self(input_tensor)
        loss = self.loss_fn(label_tensor, output["strengths"][-1])

        # print(loss)
        return loss
