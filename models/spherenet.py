import tensorflow as tf


class NormLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = tf.math.l2_normalize(strength, axis=1)
        return activation, strength


class ReluLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = tf.nn.relu(strength)
        return activation, strength


class SphereNet(tf.keras.Model):
    def __init__(self, n_hiddens, regularization=0.0, test_with_relu=False):
        super(SphereNet, self).__init__()
        assert regularization == 0.0, "Regularization is not supported!"
        print("This PerceptronNet has only 1 output!")
        self.regularizer = tf.keras.regularizers.l2(regularization)

        layer = ReluLayer if test_with_relu else NormLayer

        self.hidden_layers = [
            layer(n_hidden, kernel_regularizer=self.regularizer)
            for n_hidden in n_hiddens
        ]
        self.final_layer = layer(1, kernel_regularizer=self.regularizer)
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
        return self(input_tensor)["activations"][-1]

    def loss(self, input_tensor, label_tensor):
        output = self(input_tensor)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                label_tensor, output["strengths"][-1]
            )
        )
        print(loss)
        return loss
