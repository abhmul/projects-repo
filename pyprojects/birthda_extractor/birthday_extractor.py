import argparse
import pandas as pd
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--names", type=str, required=True)
parser.add_argument("-b", "--birthdays", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)

NAME_COLUMN = 'Subject'


def extract_birthdays(birthdays_df, names_list):
    extracted = birthdays_df[birthdays_df[NAME_COLUMN].isin(names_list)]
    diff = set(names_list) - set(extracted[NAME_COLUMN])
    return extracted, diff


if __name__ == "__main__":
    args = parser.parse_args()
    print(f'Reading names list from {args.names}')
    with open(args.names, 'r') as f:
        names = [n.replace('\n', '') for n in  list(f)]
    print(f'Reading birthdays list from {args.birthdays}')
    birthdays = pd.read_csv(args.birthdays)
    extracted_birthdays, missed_birthdays = extract_birthdays(birthdays, names)
    print()
    print('Could not find the following names:')
    pprint(missed_birthdays)
    print()
    print(f'Writing extracted birthdays to {args.output}')
    extracted_birthdays.to_csv(args.output, index=False)