import data
import functional as F
import numpy as np
from sklearn.model_selection import ParameterGrid
from mlp import MLP
from trainers import minimize_binary_label
import tensorflow as tf
import plotting
from pprint import pprint

NUM_DATA = 200
NUM_ITER = 200
PROBLEMS = {
    "blobs": {"data": data.generate_blobs(NUM_DATA), "num_iter": NUM_ITER, "batch_size": NUM_DATA},
    "circles": {"data": data.generate_circles(NUM_DATA), "num_iter": NUM_ITER, "batch_size": NUM_DATA},
    "moons": {"data": data.generate_moons(NUM_DATA), "num_iter": NUM_ITER, "batch_size": NUM_DATA},
    "xor": {"data": data.generate_xor(NUM_DATA), "num_iter": NUM_ITER, "batch_size": NUM_DATA},
}
GRID_SIZE=7
DEGREE = 2
def make_grid(degree, num):
    return {f'x{i}': np.linspace(-1.5, 1.5, num=num) for i in range(degree+1)}

def polynomial(poly):
    def apply(x):
        # print(poly)
        return sum(poly[i] * x ** i for i in range(len(poly)))
    return apply

heaveside_grad_polynomial = polynomial([1])
linear_grad_polynomial = polynomial([1])
relu_grad_polynomial = polynomial([1])

@tf.custom_gradient
def linear_grad(x):
    def grad(dy):
        # print(x)
        return linear_grad_polynomial(x) * dy

    return x, grad


class LinearLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = linear_grad(strength)
        return activation, strength

@tf.custom_gradient
def heaveside_grad(x):
    def grad(dy):
        return heaveside_grad_polynomial(x) * dy

    return F.heaveside(x), grad


class HeavesideLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = heaveside_grad(strength)
        return activation, strength

@tf.custom_gradient
def relu_grad(x):
    def grad(dy):
        return relu_grad_polynomial(x) * dy

    return tf.nn.relu(x), grad


class RiggedReluLayer(tf.keras.layers.Dense):
    def call(self, x):
        strength = super().call(x)
        activation = relu_grad(strength)
        return activation, strength

param_grid = [make_grid(DEGREE, GRID_SIZE)]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

losses_per_param = []
for param_set in ParameterGrid(param_grid):
    losses_per_problem = {}

    # Set polynomial
    heaveside_grad_polynomial = polynomial([param_set[f'x{i}'] for i in range(len(param_set))])
    print(f"\nTesting {param_set}")

    # Train model on problems and record losses
    for name, problem in PROBLEMS.items():
        print(f"Running on problem {name}")

        # Create model  
        model = MLP([20, 20], layer_constructor=HeavesideLayer)

        losses_per_problem[name] = minimize_binary_label(
            problem["data"]["train"], 
            model, 
            optimizer, 
            num_iter=problem['num_iter'], 
            batch_size=problem['batch_size'],
            log_step=problem['num_iter'] // 2
            )
        
    # Store losses
    losses_per_param.append({"params": param_set, "losses": losses_per_problem, "model": model})


def to_numpy_pred_func(pred_func):
    def new_pred_func(np_inputs):
        tensor_inputs = tf.convert_to_tensor(np_inputs)
        return pred_func(tensor_inputs).numpy()

    return new_pred_func

# Report best params per problem
for name, problem in PROBLEMS.items():
    l = sorted(range(len(losses_per_param)), key=lambda i: losses_per_param[i]["losses"][name])
    print(f"\nProblem: {name}")
    print(f"Params: {losses_per_param[l[0]]['params']}")
    print(f"Loss: {losses_per_param[l[0]]['losses'][name]}")
    X, y = data.dataset_to_numpy(problem["data"]["train"])
    plotting.plot_decision_boundary(to_numpy_pred_func(losses_per_param[l[0]]['model'].pred), X, y)
    print("Top 2-10:")
    pprint([(losses_per_param[i]['params'], losses_per_param[i]['losses'][name]) for i in l[1:10]])

# Report best params averaged over problems
l = sorted(range(len(losses_per_param)), key=lambda i: sum(losses_per_param[i]["losses"][name] for name in PROBLEMS.keys()))
print("\nAll problems")
print(f"Params: {losses_per_param[l[0]]['params']}")
for name, problem in PROBLEMS.items():
    print(f"Loss: {losses_per_param[l[0]]['losses'][name]}")
    X, y = data.dataset_to_numpy(problem["data"]["train"])
    plotting.plot_decision_boundary(to_numpy_pred_func(losses_per_param[l[0]]['model'].pred), X, y)
print("Top 2-10:")
pprint([(losses_per_param[i]['params'], losses_per_param[i]['losses']) for i in l[1:10]])



