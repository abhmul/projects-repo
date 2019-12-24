import tensorflow as tf

from ..common import functional as F


def minimize_binary_label(dataset, model, optimizer, num_iter=100, batch_size=200):
    loss_fn = lambda: model.loss(inputs, labels)  # noqa: E731
    var_list_fn = lambda: model.trainable_weights  # noqa: E731

    # setup the dataset
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size)

    for i in range(num_iter):
        for inputs, labels in dataset:
            inputs = tf.cast(inputs, dtype=tf.float32)
            labels = tf.cast(labels, dtype=tf.float32)
            # labels = F.binary_to_signed(tf.cast(labels, dtype=tf.float32))
            if labels.ndim == 1:
                labels = tf.expand_dims(labels, -1)

            print(f"{i+1}/{num_iter}")
            optimizer.minimize(loss_fn, var_list_fn)


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
