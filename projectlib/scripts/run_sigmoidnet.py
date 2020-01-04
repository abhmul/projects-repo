import tensorflow as tf

from ..common import plotting
from ..data import data
from ..models import sigmoidnet, spherenet
from ..training import trainers


# dataset = data.generate_blobs()
# dataset = data.generate_xor()
dataset = data.generate_moons()
# dataset = data.generate_circles()


model = sigmoidnet.SigmoidNet1Layer(n_hidden=100)
# model = sigmoidnet.BackpropNet1Layer(n_hidden=100)
# model = spherenet.SphereNet(n_hiddens=[100])

# Higher learning rates will converge faster, but it might not converge as well or it might diverge or bounce around
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)


trainers.minimize_binary_label(
    dataset["train"], model, optimizer, num_iter=500, batch_size=20
)

print(model.trainable_variables)


def to_numpy_pred_func(pred_func):
    def new_pred_func(np_inputs):
        tensor_inputs = tf.convert_to_tensor(np_inputs)
        return pred_func(tensor_inputs).numpy()

    return new_pred_func


X, y = data.dataset_to_numpy(dataset["train"])
plotting.plot_decision_boundary(to_numpy_pred_func(model.pred), X, y)
