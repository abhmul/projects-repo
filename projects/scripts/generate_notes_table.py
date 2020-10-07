import argparse

from projects.projectslib.py_utils import relative_date_range, print_items

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", type=int, default=0)
parser.add_argument("-e", "--end", type=int, required=True)
parser.add_argument("-n", "--columns", type=int, default=0)
parser.add_argument("--step", type=int, default=1)

DELIMITER = " | "

if __name__ == "__main__":
    args = parser.parse_args()
    dates = list(relative_date_range(args.start, args.end, args.step))
    dates.reverse()
    col_seps = [[DELIMITER] * len(dates)] * (args.columns - 1)
    table = ["".join(map(str, row)) for row in zip(dates, *col_seps)]
    print_items(table)
