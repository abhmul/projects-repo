#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser("URLify a string")
parser.add_argument("-s", "--string", default="", help="string to urlify")
args = parser.parse_args()


def urlify(s):
    return s.strip().replace(" ", "%20")


string = args.string or input()

print(urlify(string))
