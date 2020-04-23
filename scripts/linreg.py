import argparse
from typing import Tuple

from sklearn.linear_model import LinearRegression
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-x", "--x", required=True, help="x values to fit")
parser.add_argument("-y", "--y", required=True, help="y values to fit")
parser.add_argument("--no_intercept", action="store_true", help="Don't fit the intercept")


def input_to_arr(inp: str):
    return np.asarray([float(i) for i in inp.replace(' ', '').split(',')])


def parse_stdin(x: str, y: str) -> Tuple[np.ndarray, np.ndarray]:
    x = input_to_arr(x)
    y = input_to_arr(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x, y


def fit_model(x: np.ndarray, y: np.ndarray, **kwargs) -> LinearRegression:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    lr = LinearRegression(**kwargs).fit(x, y)
    return lr


if __name__ == "__main__":
    args = parser.parse_args()
    x, y = parse_stdin(args.x, args.y)
    lr = fit_model(x, y, fit_intercept=not args.no_intercept)

    # Stats
    print(f"R^2: {lr.score(x, y)}")
    print(f"Coefficients: {lr.coef_}")
    print(f"Intercept: {lr.intercept_}")



