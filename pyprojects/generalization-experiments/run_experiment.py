import argparse
import numpy as np
from pprint import pprint

from experiments import EXPERIMENTS
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--trials', type=int, default=1, help="How many trials?")
# parser.add_argument('experiments', choices=list(EXPERIMENTS.keys()), type=int, nargs="+", help="Which experiment to run?")

rng = np.random.default_rng()
# experiments = list(sorted(EXPERIMENTS.keys()))
# experiments = list(range(31, 51))
experiments = [32]

# Experiment 0 is a test experiment
# experiments.remove(0)

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running {args.trials} trial(s) of experiments {experiments}.")
    for exp in experiments:
        for _ in range(args.trials):
            result = EXPERIMENTS[exp](rng)

            pprint(result)

            with open(utils.experiment_logfile(exp), 'a') as f:
                print(result, file=f)
