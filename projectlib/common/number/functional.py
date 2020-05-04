import math
import itertools
from collections import defaultdict

flatten_iter = itertools.chain.from_iterable


# https://stackoverflow.com/a/6909532/5538273
def factors(n):
    return set(flatten_iter((i, n//i) for i in range(1, int(math.sqrt(n)+1)) if n % i == 0))


def binary(x, padding=0):
    return format(x, f"0{padding}b")


def digit_map(x: int, func=list):
    return func(str(x))

def digit_counts(x: int):
    counts = defaultdict(int)
    for i in str(x):
        counts[i] += 1
    return counts
