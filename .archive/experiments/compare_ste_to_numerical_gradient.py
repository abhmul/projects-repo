import tensorflow as tf

from projectlib.models import mlp, stemodel
from projectlib.common import plotting
from projectlib.data import data
from projectlib.training import trainers

dataset = data.generate_blobs()
# dataset = data.generate_xor()
# dataset = data.generate_moons()
# dataset = data.generate_circles()


model = mlp.MLP(n_hiddens=[10], layer_construtor=stemodel.HeavesideLayer)

# Higher learning rates will converge faster, but it might not converge as well or it might diverge or bounce around
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


trainers.minimize_binary_label(
    dataset["train"], model, optimizer, num_iter=100, batch_size=100
)


def to_numpy_pred_func(pred_func):
    def new_pred_func(np_inputs):
        tensor_inputs = tf.convert_to_tensor(np_inputs)
        return pred_func(tensor_inputs).numpy()

    return new_pred_func


X, y = data.dataset_to_numpy(dataset["train"])
plotting.plot_decision_boundary(to_numpy_pred_func(model.pred), X, y)
