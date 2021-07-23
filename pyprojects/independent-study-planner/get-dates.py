import datetime
from enum import Enum
import pyperclip


class WeekDays(Enum):
    M = 0
    T = 1
    W = 2
    R = 3
    F = 4


SEMESTER_START = datetime.date(2021, 8, 23)
SEMESTER_END = datetime.date(2021, 12, 10)

# I'll mark these by hand in the spreadsheet rather than filter them out
THANKSGIVING = (datetime.date(2021, 11, 25), datetime.date(2021, 11, 26))
LABOR_DAY = (datetime.date(2021, 9, 6), datetime.date(2021, 9, 6))

# MWF Schedule
MWF = [2, 2, 3]
# Weekly schedule
W = [4, 3]


# Inclusive
def date_range(start, end, steps):
    steps_delta = [datetime.timedelta(days=s) for s in steps]
    yield start
    d = start
    while True:
        for s in steps_delta:
            d = d + s
            if d > end:
                return
            yield d


def filter_dateranges(*dateranges):
    def func(date):
        return not any(date >= r[0] and date <= r[1] for r in dateranges)
    return func


def class_dates(start, end, steps):
    dates = date_range(start, end, steps)
    return "\n".join((d.isoformat() + "\t" + WeekDays(d.weekday()).name) for d in dates)


def week_dates(start, end):
    week_start_dates = list(date_range(start, end, [7]))
    week_end_dates = list(date_range(start + datetime.timedelta(5), end, [7]))
    if len(week_end_dates) < len(week_start_dates):
        assert len(week_end_dates) == len(week_start_dates) - 1
        week_end_dates.append(end)
    return "\n".join(f"{s}\t{e}" for s, e in zip(week_start_dates, week_end_dates))


if __name__ == "__main__":
    # filter_func = filter_dateranges(THANKSGIVING, LABOR_DAY)

    # output_string = class_dates(SEMESTER_START, SEMESTER_END, MWF)
    output_string = week_dates(SEMESTER_START, SEMESTER_END)

    print("Dates:")
    print(output_string)
    pyperclip.copy(output_string)
    print("Copied to clipboard!")

