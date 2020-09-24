from functools import cmp_to_key
from collections import defaultdict
import heapq
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("items", nargs="+")
parser.add_argument("-l", "--load", help="Path to pickle file to load saved data from.")


class Comparator:
    def __init__(self):
        self.memo = {}

    def __call__(self, a, b):
        if (a, b) in self.memo:
            pass
        elif (b, a) in self.memo:
            self.memo[a, b] = -1 * self.memo[b, a]
        else:
            self.memo[a, b] = compare(a, b)
        return self.memo[a, b]


def compare(a, b):
    choice = input(f"{a} ? {b}. Choose <, >, or =\n")
    if choice == "<":
        return -1
    elif choice == ">":
        return 1
    elif choice == "=":
        return 0
    else:
        print("Invalid selection, please select again")
        compare(a, b)


def heapsort(iterable, key=lambda x: x):
    h = []
    for value in iterable:
        heapq.heappush(h, (key(value), value))
    return [heapq.heappop(h)[1] for i in range(len(h))]


def compute_rank(sorted_list, key=lambda x: x):
    rank = 1
    rank_dict = defaultdict(list, {rank: [sorted_list[0]]})
    for val1, val2 in zip(sorted_list[:-1], sorted_list[1:]):
        if key(val1) != key(val2):
            rank += 1
        rank_dict[rank].append(val2)
    return sum(
        (
            list(zip([i] * len(rank_dict[i]), rank_dict[i]))
            for i in range(1, len(rank_dict) + 1)
        ),
        [],
    )


def close_program():
    choice = str.lower(input("Would you like to save? [Y/n]: "))
    if choice == "" or choice == "y":
        save_name = input("Save name: ")
        try:
            with open(f"{save_name}.pkl", "wb") as f:
                pickle.dump(comparator, f)
            print(f"File saved to {save_name}")
        except OSError:
            print("Unable to save file")
        exit()
    elif choice == "n":
        print("Not saving")
        exit()
    else:
        print(f"Invalid choice: {choice}")
        close_program()


if __name__ == "__main__":
    args = parser.parse_args()
    init_memo = {}
    if args.load:
        with open(args.load, "rb") as f:
            comparator = pickle.load(f)
    else:
        comparator = Comparator()

    try:
        key = cmp_to_key(comparator)
        ranked_items = compute_rank(heapsort(args.items, key=key), key=key)
    except KeyboardInterrupt:
        close_program()

    for rank, item in ranked_items:
        print(f"{rank}: {item}")

    close_program()
