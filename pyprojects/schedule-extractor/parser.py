import re
from course import *
import argparse
import pandas as pd

"""
Course blocks can be split up by occurence of '(MATH|MCS|STAT|PHYS) \d\d\d\n'. The occurence is the course number for the course block.

Format:
[Course Number] (MATH|MCS|STAT|PHYS) \d\d\d
[Title] (\w+\s)+\w+
[Hours] [Prerequisites]? Prerequisite(s): ... [Recommended]? Recommended background: ...

[Sections] This part is a table with the following fields
- CRN
- Course Type
- Start & End Time
- Meeting Days
- Room
- Building Code
- Instructor
- Meets Between
- Instructional Method

Code Outline
1. Parse out course blocks
2. Parse out items in course blocks
- course number
- title
- description
- sections
3. Parse out prerequisites and recommended background from description

We parse out groups from the main block. We write parsers for each group as necessary.
"""

argparser = argparse.ArgumentParser()
argparser.add_argument("input_type", type=InputType, choices=list(InputType))
argparser.add_argument("course_files", nargs="+", type=str, help="Files to parse for courses")
argparser.add_argument("-o", "--output", type=str, required=True, help="Filename to store output to")

def parse_file(fname: str, input_type: InputType):
    with open(fname, 'r') as course_file:
        raw_text = course_file.read()
    return parse_courses(raw_text, input_type)

def to_df(courses) -> pd.DataFrame:
    # Extract relevant data
    courses = [c.to_dict() for c in courses]
    return pd.DataFrame.from_records(courses)

if __name__ == "__main__":
    """
    take list of courses text files
    run the course parser over each and concat
    turn into pandas object
    output as csv
    """
    args = argparser.parse_args()

    courses = sum((parse_file(fname, args.input_type) for fname in args.course_files), [])
    courses_df = to_df(courses)

    courses_df.to_csv(args.output, index=False)
