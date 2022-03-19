import argparse
import numpy as np
from pprint import pprint

from experiments import EXPERIMENTS
from experimenter import Experimenter
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--trials', type=int, default=1, help="How many trials?")
parser.add_argument('-tb', '--tensorboard', action='store_true', help="Should we write to TensorBoard")

rng = np.random.default_rng()
experiments = [0]
# experiments = list(sorted(EXPERIMENTS.keys()))
# experiments = list(range(31, 51))

# Experiment 0 is a test experiment
# experiments.remove(0)

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running {args.trials} trial(s) of experiments {experiments}.")
    exp_runner = Experimenter(write_to_tensorboard=args.tensorboard)
    for exp in experiments:
        for _ in range(args.trials):
            exp_runner.run_experiment(exp, rng)