import argparse
from functools import wraps, cmp_to_key
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('to_sort', nargs="+", help="list to relsort")


def cached(func):
    func.cache = {}

    @wraps(func)
    def wrapper(*args):
        try:
            return func.cache[args]
        except KeyError:
            func.cache[args] = result = func(*args)
            return result
    return wrapper


@cached
def _compare(a, b):
    print(f"\n{a} or {b}?")
    res = int(input(f"'1' for {a} and '2' for {b} and '0' if equal: "))
    if res == 1:
        print(f"{a} < {b}")
        return -1
    elif res == 2:
        print(f"{a} > {b}")
        return 1
    else:
        print(f"{a} = {b}")
        return 0


def compare(a, b):
    if (b, a) in _compare.cache:
        return -1 * _compare(b, a)
    else:
        return _compare(a, b)


if __name__ == "__main__":
    args = parser.parse_args()
    ranked = sorted(args.to_sort, key=cmp_to_key(compare))
    pprint(ranked)
    ranks = {}
    i = 1
    ranks[ranked[0]] = 1
    for item1, item2 in zip(ranked[:-1], ranked[1:]):
        res = compare(item1, item2)
        if res == 0:
            ranks[item2] = ranks[item1]
        else:
            i += 1
            ranks[item2] = i

    print("Ranks")
    for item in ranked:
        print(f"{ranks[item]}. {item}")
