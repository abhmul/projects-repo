from typing import Dict
from models import Loss, Model

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

# @tf.function
def train_step(model_with_loss: Loss, optimizer: Optimizer, x_batch, y_batch) -> Dict[str, tf.Tensor]:
    with tf.GradientTape(persistent=True) as tape: # type: ignore
        _ = model_with_loss(x_batch, y_batch)
        loss_value = model_with_loss.metric_values["_loss"]
        # print('watched variables')
        # print(tape.watched_variables())
        loss_mean = tf.reduce_mean(loss_value)
        # encoded_mean = tf.reduce_mean(tape.watched_variables()[-1])

    metric_values = model_with_loss.metric_values
    error = tf.cast(tf.argmax(metric_values["outputs"], axis=1) != tf.argmax(y_batch, axis=1), tf.float32)
    metric_values["error"] = error
    model_with_loss.reset()
    # grads = tape.gradient(tf.reduce_mean(loss_value), self.encoder_decoder.parameters())
    # print('Encoder Decoder Params')
    # print(self.encoder_decoder.parameters())
    # print(tape.gradient(tf.reduce_mean(loss_value), tape.watched_variables()))
    # print('dLoss / dWatched')
    # print(tape.gradient(loss_mean, tape.watched_variables()))
    # for var in tape.watched_variables():
    #     print(f'd{var.name}/dWatched')
    #     print(tape.gradient(var, tape.watched_variables()))
    
    # grads = None
    # print(grads)
    grads = tape.gradient(loss_mean, model_with_loss.parameters())
    optimizer.apply_gradients(zip(grads, model_with_loss.parameters()))
    return metric_values

# @tf.function
# def test_step(model_with_loss: Loss, x_batch, y_batch) -> Dict[str, tf.Tensor]:
#     _ = model_with_loss(x_batch, y_batch)
#     metric_values = model_with_loss.metric_values
#     model_with_loss.reset()
#     return metric_values

@tf.function
def test_step(model: Model, x_batch) -> Dict[str, tf.Tensor]:
    _ = model(x_batch)
    metric_values = model.metric_values
    model.reset()
    return metric_values