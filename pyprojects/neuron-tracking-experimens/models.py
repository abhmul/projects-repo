from typing import Any, Dict, Tuple
import abc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def norm(x):
    return x / tf.math.l2_normalize(x, axis=1)

INITIALIZERS = {
    "he_normal": keras.initializers.HeNormal()
}
ACTIVATIONS = {
    "relu": keras.activations.relu,
    "sphere": norm,
    "linear": lambda x: x
}

class Model():

    def __init__(self) -> None:
        self._metric_values: Dict[str, tf.Tensor] = {}

    def update_metric(self, metric_dict: Dict[str, tf.Tensor]):
        self._metric_values.update(metric_dict)
    
    def __call__(self, *args: Any, **kwds: Any) -> tf.Tensor:
        return self.call(*args, **kwds)
    
    @abc.abstractmethod
    def call(self, *args: Any, **kwds: Any) -> tf.Tensor:
        pass
    
    @property
    def metric_values(self) -> Dict[str, tf.Tensor]:
        return dict(self._metric_values)
    
    def parameters(self):
        return []

    def reset(self):
        self._metric_values = {}
    

class Loss(Model):
    
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self._metric_values = self.model._metric_values  # Inherit the metric values of the underlying model
    
    @abc.abstractmethod
    def call(self, inputs, target):
        pass
    
    def parameters(self):
        return self.model.parameters()

class BinaryCrossEntropyWithLogits(Loss):
    
    def call(self, inputs, target):
        outputs = self.model(inputs)
        self.update_metric({"outputs": outputs})
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=outputs)
        self.update_metric({"_loss": loss})
        return loss

class CategoricalCrossEntropyWithLogits(Loss):
    
    def call(self, inputs, target):
        outputs = self.model(inputs)
        self.update_metric({"outputs": outputs})
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=outputs)
        self.update_metric({"_loss": loss})
        return loss

class MLP(Model):
    
    def __init__(self, outputs, hidden, depth=1, initializer="he_normal", activation='relu') -> None:
        super().__init__()
        assert depth == 1
        
        self.initializer = initializer
        self.hidden_layer = layers.Dense(
            hidden, 
            activation="linear",
            name="hidden_layer",
            kernel_initializer=INITIALIZERS[initializer],
            bias_initializer="zeros"
        )
        self.output_layer = layers.Dense(
            outputs,
            activation="linear",
            name='output_layer',
            kernel_initializer=INITIALIZERS[initializer],
            bias_initializer="zeros"
        )
        self.activation = activation
        self.activation_func = ACTIVATIONS[activation]
    
    def call(self, inputs):
        x = self.hidden_layer(inputs)
        self.update_metric({"hidden_strength": x})
        x = self.activation_func(x)
        self.update_metric({"hidden_activation": x})
        outputs = self.output_layer(x)
        return outputs
    
    def parameters(self):
        return self.hidden_layer.weights + self.output_layer.weights
