from asyncore import write
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from projectslib.data import load_mnist, mix_classification_datasets, sample, retrieve_split
from tqdm import trange

from data import mnist_mlp_connector
from models import MLP, CategoricalCrossEntropyWithLogits
from trainer import train_step, test_step

EXPERIMENTS = {}


def experiment(exp_number, description):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Running experiment {exp_number}")
            print(description)
            return func(*args, **kwargs)

        EXPERIMENTS[exp_number] = wrapper

        return wrapper
    return decorator

def mlp_mnist_neuron_tracking_experiment(rng, logger, summary_writer, inspection_sample_size, hidden_size, initializer, learning_rate, momentum, nesterov, steps, batch_size):
    X, Y = mix_classification_datasets(*mnist_mlp_connector(load_mnist()))
    _, _, inds = sample((X, Y), inspection_sample_size, rng, replace=False)
    print(f"Sampled {len(inds)} datapoints")
    
    print(np.mean(X))
    
    (x_inspect, y_inspect), (Xtr, Ytr) = retrieve_split((X, Y), inds)
    assert len(x_inspect) == len(y_inspect)
    x_inspect_tf = tf.convert_to_tensor(x_inspect)
    y_inspect_tf = tf.convert_to_tensor(y_inspect)
    y_inspect_class = np.argmax(y_inspect, axis=1)
    hidden_strengths_trials = []
    hidden_activations_trials = []
    trials = 5
    for t in range(trials):
        model = MLP(Y.shape[1], depth=1, hidden=hidden_size, initializer=initializer, activation='linear')
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        model_with_loss = CategoricalCrossEntropyWithLogits(model)

        print(f"Training model for {steps} steps and with {batch_size} batch size.")
        hidden_strengths = []
        hidden_activations = []
        for i in range(steps):
            print(f"starting step {i+1}")
            x_batch, y_batch, batch_inds = sample((Xtr, Ytr), batch_size, rng, replace=True)
            metric_values = train_step(model_with_loss, opt, tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch))
            
            logger.log(f"Step {i+1}/{steps} : loss - {tf.reduce_mean(metric_values['_loss']).numpy()} : Error - {tf.reduce_mean(metric_values['error']).numpy()}")
            
            # metric_values = test_step(model_with_loss, x_inspect_tf, y_inspect_tf)
            # metric_values = test_step(model, x_inspect_tf)
            _ = model(x_inspect_tf)
            metric_values = model.metric_values
            model.reset()
            # if summary_writer is not None:
            #     hidden_strengths.append(metric_values["hidden_strength"])
            #     hidden_activations.append(metric_values["hidden_activation"])
            hidden_strengths.append(metric_values["hidden_strength"].numpy())
            hidden_activations.append(metric_values["hidden_activation"].numpy())
        
        hidden_strengths = np.stack(hidden_strengths, axis=0)
        hidden_strengths_trials.append(hidden_strengths)
        hidden_activations = np.stack(hidden_activations, axis=0)
        hidden_activations_trials.append(hidden_activations)
    
    hidden_strengths_trials = np.stack(hidden_strengths_trials, axis=0)
    hidden_activations_trials = np.stack(hidden_activations_trials, axis=0)
    x_axis_data = np.arange(steps)
    # y_axis_data = {j: {f"example_{j}_class_{y_inspect_class[j]}_hidden_str_{hi}": hidden_strengths[:, j, hi] for  hi in range(hidden_size) } for j in range(y_inspect_class.shape[0]) }
    # for inspect_num in y_axis_data:
    #     for datum_name in y_axis_data[inspect_num]:
    #         plt.plot(x_axis_data, y_axis_data[inspect_num][datum_name], label=datum_name)
    #     plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    #     plt.show()
    y_axis_data = {j: {hi: {f"ex{j}_cls{y_inspect_class[j]}_hi{hi}_tr{t}": hidden_strengths_trials[t, :, j, hi] - hidden_strengths_trials[t, 0, j, hi] for t in range(trials)} for hi in range(hidden_size)} for j in range(y_inspect_class.shape[0])}
    # y_axis_data = {j: {hi: {f"ex{j}_cls{y_inspect_class[j]}_hi{hi}_tr{t}": hidden_activations_trials[t, :, j, hi] for t in range(trials)} for hi in range(hidden_size)} for j in range(y_inspect_class.shape[0])}
    for inspect in y_axis_data.values():
        for hid in inspect.values():
            for name, tr in hid.items():
                plt.plot(x_axis_data, tr, label=name)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
    
    # if summary_writer is not None:
        # write_to_tensorboard(summary_writer, hidden_strengths, hidden_activations, hidden_size, y_inspect_class)

# @tf.function
# def write_to_tensorboard(summary_writer, hidden_strengths, hidden_activations, hidden_size, y_inspect_class):
#     print(type(hidden_strengths[0]))
#     with summary_writer.as_default():
#         for i in trange(len(hidden_activations)):
#             for hi in range(hidden_size):
#                 for j in range(y_inspect_class.shape[0]):
#                     name = f"example_{j}_class_{y_inspect_class[j]}_hidden_str_{hi}"
#                     tf.summary.scalar(name, hidden_strengths[i][j, hi], step=i)
#                     name = f"example_{j}_class_{y_inspect_class[j]}_hidden_act_{hi}"
#                     tf.summary.scalar(name, hidden_activations[i][j, hi], step=i)       

@experiment(
   0, "TEST"
)
def experiment0(rng, logger, summary_writer):
    return mlp_mnist_neuron_tracking_experiment(rng, logger, summary_writer, 50, 30, "he_normal", 1e-2, 0, False, 500, 32)


# @experiment(
#     1,
#     """
#     MNIST 1k
#     MLP 1 32 HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment1(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     2,
#     """
#     MNIST 1k
#     MLP 2 32 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment2(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     3,
#     """
#     MNIST 1k
#     MLP 3 32 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment3(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     4,
#     """
#     MNIST 1k
#     MLP 4 32 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment4(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     5,
#     """
#     MNIST 1k
#     MLP 5 32 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment5(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=32, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# # Same as experiment 1
# # @experiment(
# #     6,
# #     """
# #     MNIST 1k
# #     MLP 1 64 HeNormal
# #     Vanilla SGD 1e3
# #     CCE
# #     ep 30
# #     batch_size 32
# #     """
# # )


# @experiment(
#     7,
#     """
#     MNIST 1k
#     MLP 2 64 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment7(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     8,
#     """
#     MNIST 1k
#     MLP 3 64 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment8(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     9,
#     """
#     MNIST 1k
#     MLP 4 64 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment9(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     10,
#     """
#     MNIST 1k
#     MLP 5 64 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment10(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=64, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# # Same as experiment 1
# # @experiment(
# #     11,
# #     """
# #     MNIST 1k
# #     MLP 1 128 HeNormal
# #     Vanilla SGD 1e3
# #     CCE
# #     ep 30
# #     batch_size 32
# #     """
# # )

# @experiment(
#     12,
#     """
#     MNIST 1k
#     MLP 2 128 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment12(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     13,
#     """
#     MNIST 1k
#     MLP 3 128 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment13(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     14,
#     """
#     MNIST 1k
#     MLP 4 128 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment14(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     15,
#     """
#     MNIST 1k
#     MLP 5 128 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment15(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=128, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     16,
#     """
#     MNIST 1k
#     MLP 1 256 HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment16(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     17,
#     """
#     MNIST 1k
#     MLP 2 256 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment17(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     18,
#     """
#     MNIST 1k
#     MLP 3 256 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment18(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     19,
#     """
#     MNIST 1k
#     MLP 4 256 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment19(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     20,
#     """
#     MNIST 1k
#     MLP 5 256 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment20(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=256, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# # Same as experiment 1
# # @experiment(
# #     21,
# #     """
# #     MNIST 1k
# #     MLP 1 512 HeNormal
# #     Vanilla SGD 1e3
# #     CCE
# #     ep 30
# #     batch_size 32
# #     """
# # )

# @experiment(
#     22,
#     """
#     MNIST 1k
#     MLP 2 512 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment22(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     23,
#     """
#     MNIST 1k
#     MLP 3 512 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment23(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     24,
#     """
#     MNIST 1k
#     MLP 4 512 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment24(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     25,
#     """
#     MNIST 1k
#     MLP 5 512 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment25(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=512, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# # Same as experiment 1
# # @experiment(
# #     26,
# #     """
# #     MNIST 1k
# #     MLP 1 1024 HeNormal
# #     Vanilla SGD 1e3
# #     CCE
# #     ep 30
# #     batch_size 32
# #     """
# # )


# @experiment(
#     27,
#     """
#     MNIST 1k
#     MLP 2 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment27(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     28,
#     """
#     MNIST 1k
#     MLP 3 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment28(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     29,
#     """
#     MNIST 1k
#     MLP 4 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment29(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     30,
#     """
#     MNIST 1k
#     MLP 5 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment30(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=1000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     31,
#     """
#     MNIST 5k
#     MLP 1 N/A HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment31(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     32,
#     """
#     MNIST 5k
#     MLP 2 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment32(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     33,
#     """
#     MNIST 5k
#     MLP 3 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment33(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     34,
#     """
#     MNIST 5k
#     MLP 4 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment34(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     35,
#     """
#     MNIST 5k
#     MLP 5 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment35(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=5000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     36,
#     """
#     MNIST 10k
#     MLP 1 NA HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment36(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     37,
#     """
#     MNIST 10k
#     MLP 2 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment37(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     38,
#     """
#     MNIST 10k
#     MLP 3 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment38(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     39,
#     """
#     MNIST 10k
#     MLP 4 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment39(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     40,
#     """
#     MNIST 10k
#     MLP 5 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment40(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=10000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     41,
#     """
#     MNIST 5k
#     MLP 1 N/A HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment41(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     42,
#     """
#     MNIST 5k
#     MLP 2 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment42(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     43,
#     """
#     MNIST 5k
#     MLP 3 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment43(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     44,
#     """
#     MNIST 5k
#     MLP 4 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment44(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     45,
#     """
#     MNIST 5k
#     MLP 5 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment45(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=15000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     46,
#     """
#     MNIST 20k
#     MLP 1 NA HeNormal
#     Vanilla SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment46(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1, depth=1, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     47,
#     """
#     MNIST 20k
#     MLP 2 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment47(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=2, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)


# @experiment(
#     48,
#     """
#     MNIST 20k
#     MLP 3 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment48(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=3, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     49,
#     """
#     MNIST 20k
#     MLP 4 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment49(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=4, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)

# @experiment(
#     50,
#     """
#     MNIST 20k
#     MLP 5 1024 HeNormal
#     SGD 1e3
#     CCE
#     ep 30
#     batch_size 32
#     """
# )
# def experiment50(rng):
#     return mlp_mnist_sgd_experiment(rng, sample_size=20000, hidden_size=1024, depth=5, initializer="he_normal", learning_rate=1e-3, momentum=0.0, nesterov=False, epochs=30, batch_size=32)