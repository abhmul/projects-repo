from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pprint import pprint
import math
from pathlib import Path

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
    "--degree_list", nargs="+", type=int, default=[], help="list of degrees for model"
)

bp2_to_numeric = {"D": 0, "D?": 0.25, "H?": 0.75, "H": 1.0}

PREDICT = "bp2_classification"
PREDICT_RANGE = (0.0, 1.0)


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


def compile_wavelet(i, ai, bi, data_size, tau=1):
    def wavelet(x):
        return ai * cos(i * 2 * math.pi / (tau * data_size) * x) + bi * sin(
            i * 2 * math.pi / (tau * data_size) * x
        )

    return wavelet


def fourier_series(x, data_size, tau=1, degree_list=tuple()):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    (a0,) = parameters("a0")
    cos_a = parameters(",".join(["a{}".format(i) for i in degree_list]))
    sin_b = parameters(",".join(["b{}".format(i) for i in degree_list]))
    # Construct the series
    series = a0 + sum(
        ai * cos(i * 2 * math.pi / (tau * data_size) * x)
        + bi * sin(i * 2 * math.pi / (tau * data_size) * x)
        for (i, ai, bi) in zip(degree_list, cos_a, sin_b)
    )
    return series


def compile_model(degree_list, data_size):
    x, y = variables("x, y")
    # (w,) = parameters("w")
    model_dict = {y: fourier_series(x, data_size, tau=1, degree_list=degree_list)}

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
    a = zip([f"a{i}" for i in range(1, len(a) + 1)], a)
    b = zip([f"b{i}" for i in range(1, len(b) + 1)], b)
    return list(a) + list(b)


if __name__ == "__main__":
    args = parser.parse_args()

    mood_data = load_data(args.data, date_range=(args.start, args.end))
    mood_data = extract_data(mood_data)
    degrees = args.degree_list if args.degree_list else range(1, args.degree + 1)
    model_dict = compile_model(degrees, max(mood_data.index))

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
    out[::-1].to_csv(args.data.parent / "bp2_pred.csv", index=False)
