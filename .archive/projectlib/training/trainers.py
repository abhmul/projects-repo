import tensorflow as tf
import numpy as np

from ..common import functional as F


def get_numerical_grads(model, inputs, labels, gradients):
    epsilons = [0.001, 0.01, 0.1, 1.0, 10.0]
    for ep in epsilons:
        print(f"\n=====EPSILON={ep}======")
        for var in model.trainable_variables:
            dlosses = np.zeros(var.shape)
            for ind in np.ndindex(*var.shape):
                old_var = var
                var.assign(tf.tensor_scatter_nd_add(old_var, [list(ind)], [ep]))
                upper_loss = model.loss(labels, inputs)
                var.assign(tf.tensor_scatter_nd_add(old_var, [list(ind)], [-ep]))
                lower_loss = model.loss(labels, inputs)

                dlosses[ind] = ((upper_loss - lower_loss) / (ep)).numpy()
            print(var.name)
            print(dlosses)
        input()


def minimize_binary_label(dataset, model, optimizer, num_iter=100, batch_size=200):
    # setup the dataset
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size)

    for i in range(num_iter):
        print(f"{i+1}/{num_iter}")
        for inputs, labels in dataset:
            inputs = tf.cast(inputs, dtype=tf.float32)
            labels = tf.cast(labels, dtype=tf.float32)
            if labels.ndim == 1:
                labels = tf.expand_dims(labels, -1)

            with tf.GradientTape() as tape:
                loss_val = model.loss(labels, inputs)
                # loss_dict = model.loss(labels, inputs)
                # iter_switch = (i // 10) % 2
                # loss_val = (
                #     loss_dict["loss_final"] if iter_switch else loss_dict["loss_hidden"]
                # )
                # print(loss_val)

            gradients = tape.gradient(loss_val, model.trainable_variables)
            print("\n\n=========GRADIENTS==========")
            for i, var in enumerate(model.trainable_variables):
                print(var.name)
                print(gradients[i])
            get_numerical_grads(model, inputs, labels)

            optimizer.apply_gradients(
                [
                    (g, w)
                    for g, w in zip(gradients, model.trainable_variables)
                    if g is not None
                ]
            )


def train_model_with_updater_online(
    dataset, model, learning_rate=1.0, num_iter=100, batch_size=1
):
    # setup the dataset
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)

    for i in range(num_iter):
        costs = []
        for inputs, labels in dataset:
            inputs = tf.cast(inputs, dtype=tf.float32)
            labels = F.binary_to_signed(tf.cast(labels, dtype=tf.float32))
            if labels.ndim == 1:
                labels = tf.expand_dims(labels, -1)

            # print("===================================")
            # print("Inputs:", inputs)
            # print("Labels:", labels)

            cost = model.update(inputs, labels, learning_rate=learning_rate)
            costs.append(cost)
            # print(f"{i+1}/{num_iter}: {cost}")
            # input()
        print(f"{i+1}/{num_iter}: {sum(costs) / len(costs)}")
