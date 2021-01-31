import argparse
import random
import math


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_problems", required=True, type=int, help="number of problems to choose from")
parser.add_argument("-k", "--select", required=True, type=float, help="fraction of problems to select")
parser.add_argument("--test", default=0.5, help="fraction of problems to use on test")
parser.add_argument("--required", type=int, nargs="*", help="Questions that must be done.")
parser.add_argument("--avoid", type=int, nargs="*", help="Questions that should not be included.")

args = parser.parse_args()

problems: set = set(args.required)
not_problems: set = set(args.avoid)
num_remaining = args.num_problems - len(not_problems)

num_assigned: int = min(math.ceil(args.select * args.num_problems), num_remaining)
num_select: int = num_assigned - len(problems)

candidates: set = set(range(1, args.num_problems + 1)) - problems - not_problems
problems |= set(random.sample(candidates, num_select))
assert len(problems) == num_assigned

# Split into test and train
num_test = math.floor(args.test * len(problems))
test_problems = set(random.sample(problems, num_test))
train_problems = problems - test_problems

print(f"Train: {sorted(train_problems)}")
print(f"Test: {sorted(test_problems)}")
