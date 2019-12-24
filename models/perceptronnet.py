import tensorflow as tf

from ..common import assertions, functional as F


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = F.heaveside(strength)
        return activation, strength


class SignLayer(tf.keras.layers.Dense):
    # def build(self, input_shape):
    #     super().build(input_shape)
    #     self.kernel.assign(F.sign(tf.random_uniform_initializer()(self.kernel.shape)))
    #     self.bias.assign(tf.zeros_initializer()(self.bias.shape))

    def call(self, x):
        strength = super().call(x)
        activation = F.sign(strength)
        return activation, strength


class PerceptronNet(tf.keras.Model):
    def __init__(self, n_hiddens, regularization=0.0):
        super(PerceptronNet, self).__init__()
        assert regularization == 0.0, "Regularization is not supported!"
        print("This PerceptronNet has only 1 output!")
        self.regularizer = tf.keras.regularizers.l2(regularization)
        self.hidden_layers = [
            SignLayer(n_hidden, kernel_regularizer=self.regularizer)
            for n_hidden in n_hiddens
        ]
        self.final_layer = SignLayer(1, kernel_regularizer=self.regularizer)
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

    @staticmethod
    def error_vector(prediction, label_tensor):
        assertions.assert_is_signs(label_tensor)
        assertions.assert_is_signs(prediction)
        return (label_tensor - prediction) / 2

    @staticmethod
    def compute_previous_expected(prediction, expected, layer_input, kernel, error):
        # Compute the new label
        offset = (expected - prediction) / 2  # B x O
        assertions.assert_is_signs_or_zero(offset)

        # Kernel is I x O, this could be 0 in some places
        partial_expected = error * tf.sign(tf.matmul(offset, tf.transpose(kernel)))
        mask = tf.cast(tf.not_equal(partial_expected, 0), tf.float32)
        return mask * partial_expected + (1 - mask) * layer_input

    @staticmethod
    def compute_weight_update(prediction, expected, layer_input):
        batch_size = prediction.shape[0]
        assert batch_size == expected.shape[0] == layer_input.shape[0]

        offset = (expected - prediction) / 2
        assertions.assert_is_signs_or_zero(offset)

        return tf.matmul(tf.transpose(layer_input), offset) / batch_size

    @staticmethod
    def compute_bias_update(prediction, expected):
        batch_size = prediction.shape[0]
        assert batch_size == expected.shape[0]

        offset = (expected - prediction) / 2
        assertions.assert_is_signs_or_zero(offset)

        return tf.reduce_mean(offset, 0)

    @staticmethod
    def cost(strength, expected):
        assertions.assert_is_signs(expected)
        per_sample_cost = (tf.abs(strength) - strength * expected) / 2
        return tf.reduce_mean(per_sample_cost)

    def update(self, input_tensor, label_tensor, learning_rate=1.0):
        output = self(input_tensor)  # B x 1
        activations = output["activations"]
        prediction = activations[-1]  # B x 1
        layer_inputs = [input_tensor] + activations[:-1]
        num_layers = len(self.all_layers)

        # Get the internal labels
        offset = (label_tensor - prediction) / 2
        assertions.assert_is_signs_or_zero(offset)
        error = tf.math.abs(offset)  # B x 1
        expecteds = [label_tensor]
        # Runs over all layers except the first
        for i in range(num_layers - 1, 0, -1):
            previous_expected = self.compute_previous_expected(
                activations[i],  # B x O
                expecteds[-1],  # B x O
                layer_inputs[i],  # B x I
                self.all_layers[i].kernel,  # I x O
                error,  # B x 1
            )
            expecteds.append(previous_expected)
        expecteds.reverse()

        # Compute the weight updates
        weight_updates = []
        for i in range(num_layers - 1, -1, -1):
            weight_update = self.compute_weight_update(
                activations[i],  # B x O
                expecteds[i],  # B x O
                layer_inputs[i],  # B x I
            )
            weight_updates.append(weight_update)
        weight_updates.reverse()

        # Compute bias updates
        bias_updates = []
        for i in range(num_layers - 1, -1, -1):
            bias_update = self.compute_bias_update(
                activations[i], expecteds[i]  # B x O
            )
            bias_updates.append(bias_update)
        bias_updates.reverse()

        # Update the layers
        for i, layer in enumerate(self.all_layers):
            layer.kernel.assign_add(learning_rate * tf.stop_gradient(weight_updates[i]))
            layer.bias.assign_add(learning_rate * tf.stop_gradient(bias_updates[i]))

        # print("Error:", error)
        # print("Weight Updates:", weight_updates)
        # print("Bias Updates:", bias_updates)
        # print("Final Weights:")
        # print(self.final_layer.kernel)
        # print("Final Bias:")
        # print(self.final_layer.bias)

        return self.cost(output["strengths"][-1], label_tensor)
