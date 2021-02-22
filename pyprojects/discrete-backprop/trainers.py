import tensorflow as tf
import numpy as np

def minimize_binary_label(dataset, model, optimizer, num_iter=100, batch_size=200, log_step=5):
    # setup the dataset
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size)

    for i in range(num_iter):
        # print(f"{i+1}/{num_iter}")
        for inputs, labels in dataset:
            inputs = tf.cast(inputs, dtype=tf.float32)
            labels = tf.cast(labels, dtype=tf.float32)
            if labels.ndim == 1:
                labels = tf.expand_dims(labels, -1)

            with tf.GradientTape() as tape:
                loss_val = model.loss(labels, inputs)
                if i % log_step == 0:
                    print(f"iter {i+1} - loss: {loss_val}")

            gradients = tape.gradient(loss_val, model.trainable_variables)
            # print("\n\n=========GRADIENTS==========")
            # for i, var in enumerate(model.trainable_variables):
            #     print(var.name)
            #     print(gradients[i])
            # get_numerical_grads(model, inputs, labels)

            optimizer.apply_gradients(
                [
                    (g, w)
                    for g, w in zip(gradients, model.trainable_variables)
                    if g is not None
                ]
            )

    loss = model.loss(labels, inputs)
    print(f"iter {num_iter} - loss: {loss}")
    return loss