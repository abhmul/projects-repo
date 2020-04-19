import numpy as np

import projectlib.common.distributions as dist

RNG = np.random.default_rng(12345)

def tightest_fit(true_points):
    # points are Samples x Shape
    bound_max = np.max(true_points, axis=0)
    bound_min = np.min(true_points, axis=0)
    return bound_max, bound_min


def error(rectangles, points):
    corners_max, corners_min = rectangles
    true_point_mask = np.all(points[None] <= corners_max[:, None] &
                             points[None] >= corners_min[:, None],
                             axis=tuple(range(2, points.ndim + 1)))

    
    fit_max = np.maximum.accumulate(points, axis=0)
    fit_min = np.minimum.accumulate(points, axis=0)



def run_trial(distribution, num_games=1000, num_points=1000):
    rectangles = dist.random_rectangle(distribution.shape, samples=num_games, rng=RNG)
    points = dist.random_grid_point(distribution, samples=num_points, rng=RNG)

