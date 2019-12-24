import tensorflow as tf

from ..common import plotting
from ..data import data
from ..models import perceptronnet
from ..training import trainers


# dataset = data.generate_blobs()
# dataset = data.generate_xor()
dataset = data.generate_moons()
# dataset = data.generate_circles()

model = perceptronnet.PerceptronNet(n_hiddens=[100])


trainers.train_model_with_updater_online(
    dataset["train"], model, learning_rate=0.01, num_iter=1000, batch_size=100
)


def to_numpy_pred_func(pred_func):
    def new_pred_func(np_inputs):
        tensor_inputs = tf.convert_to_tensor(np_inputs)
        return pred_func(tensor_inputs).numpy()

    return new_pred_func


X, y = data.dataset_to_numpy(dataset["train"])
plotting.plot_decision_boundary(to_numpy_pred_func(model.pred), X, y)
