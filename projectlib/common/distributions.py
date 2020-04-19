import numpy as np
from . import functional as F


def random_pmf(shape, samples=1, rng=np.random.default_rng()):
    return rng.dirichlet(np.ones(shape).flatten(), size=samples).reshape(shape)


def random_grid_point(pvals, samples=1, rng=np.random.default_rng()):
    num_points = np.product(pvals.shape)
    flat_points = rng.choice(num_points, size=samples, p=pvals.flatten())
    return F.np_unravel(flat_points, pvals.shape)


def random_rectangle(shape, samples=1, rng=np.random.default_rng()):
    num_points = np.product(shape)
    # Samples x Shape
    corners1 = F.np_unravel(rng.integers(num_points, size=samples), shape)
    corners2 = F.np_unravel(rng.integers(num_points, size=samples), shape)
    return np.maximum(corners1, corners2), np.minimum(corners1, corners2)
