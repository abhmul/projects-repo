from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pprint import pprint
import math
from pathlib import Path

from projects.projectslib.py_utils import range2

parser = argparse.ArgumentParser()
parser.add_argument(
    "data",
    type=Path,
    default=Path(__file__).absolute().parent,
    help="path to tsv mood file",
)
parser.add_argument("-s", "--start", default=None, help="start date of selection")
parser.add_argument("-e", "--end", default=None, help="end date of selection")
parser.add_argument("-d", "--degree", type=int, default=3, help="degree of model")
parser.add_argument(
    "--degrees", nargs="+", type=float, default=[], help="list of degrees for model"
)
parser.add_argument(
    "--relative_degree", action="store_true", help="list of degrees for model"
)
parser.add_argument(
    "--periods",
    nargs="+",
    type=float,
    default=[],
    help="list of frequencies for model.",
)

bp2_to_numeric = {"D": 0, "D?": 0.25, "H?": 0.75, "H": 1.0}

PREDICT = "general"
PREDICT_RANGE = (0.0, 10.0)


def in_range(start, end):
    if start is not None and end is not None:
        return lambda x: start <= x < end
    if start is not None and end is None:
        return lambda x: start <= x
    if start is None and end is not None:
        return lambda x: x < end
    else:
        return lambda x: True


def load_data(path, date_range=(None, None)):
    data = pd.read_csv(path, sep="\t")
    data["date"] = pd.to_datetime(data["date"])
    date_range = map(pd.to_datetime, date_range)
    data = data[data["date"].map(in_range(*date_range))]
    data = data.sort_values(by="date").reset_index()
    data["notes"] = data["notes"].fillna("").astype(str)
    data = data.replace({"bp2_classification": bp2_to_numeric})
    return data


def extract_data(df):
    df = df[df[PREDICT].notna()]
    return df


def fourier_series(x, period_list):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    (a0,) = parameters("a0")
    cos_a = parameters(",".join(["a{}".format(i) for i in range2(len(period_list))]))
    sin_b = parameters(",".join(["b{}".format(i) for i in range2(len(period_list))]))
    # Construct the series
    print(period_list)
    series = a0 + sum(
        ai * cos(2 * math.pi / tau * x) + bi * sin(2 * math.pi / tau * x)
        for (tau, ai, bi) in zip(period_list, cos_a, sin_b)
    )
    return series


def build_period_list(n=None, degree_list=tuple(), period_list=tuple(), data_size=1):
    if period_list:
        return [data_size * p for p in period_list]
    if degree_list:
        return [data_size * 1 / d for d in degree_list]
    if n is not None:
        return [data_size * 1 / i for i in range2(n)]


def compile_model(period_list):
    x, y = variables("x, y")
    # (w,) = parameters("w")
    model_dict = {y: fourier_series(x, period_list)}

    print("\nModel:")
    print("==============")
    pprint(model_dict)

    return model_dict


def fit_model(model_dict, x, y):
    fit = Fit(model_dict, x=x, y=y)
    fit_result = fit.execute()

    print("\nFit Result:")
    print("==============")
    print(fit_result)

    print("Sorted coefficients:")
    print("=============")
    params = extract_parameters(fit_result)
    pprint(sorted(params, key=lambda item: abs(item[1]), reverse=True))

    return fit, fit_result


def predict(x, params, model):
    return np.clip(model(x=x, **params).y, *PREDICT_RANGE)


def extract_parameters(fit_result):
    a0 = fit_result._popt[0]
    num_a = (len(fit_result._popt) - 1) // 2
    a = fit_result._popt[1 : num_a + 1]
    b = fit_result._popt[num_a + 1 :]
    a = zip([f"a{i}" for i in range2(len(a))], a)
    b = zip([f"b{i}" for i in range2(len(b))], b)
    return list(a) + list(b)


if __name__ == "__main__":
    args = parser.parse_args()

    mood_data = load_data(args.data, date_range=(args.start, args.end))
    mood_data = extract_data(mood_data)
    data_size = max(mood_data.index) if args.relative_degree else 1
    print(data_size)
    period_list = build_period_list(
        n=args.degree,
        degree_list=args.degrees,
        period_list=args.periods,
        data_size=data_size,
    )
    model_dict = compile_model(period_list)

    xdata, ydata = mood_data.index.values, mood_data[PREDICT].values
    fit, fit_result = fit_model(model_dict, xdata, ydata)

    # Plot the result
    plt.plot(xdata, ydata)
    forecast_len = 56
    xforecast = np.arange(np.amax(xdata) + forecast_len)
    yprediction = predict(x=xforecast, params=fit_result.params, model=fit.model)
    plt.plot(
        xforecast,
        yprediction,
        color="green",
        ls=":",
    )
    plt.show()

    forecast_range = pd.Series(
        pd.date_range(mood_data["date"].iloc[-1], periods=forecast_len + 1)[1:]
    )
    date_forecast = pd.date_range(mood_data["date"].iloc[0], periods=len(xforecast))
    out = pd.DataFrame({"date": date_forecast, "prediction": yprediction})
    out[::-1].to_csv(args.data.parent / "mood_pred.csv", index=False)
