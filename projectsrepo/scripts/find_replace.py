import re

if __name__ == "__main__":
    original_string = input("Original String: ")
    print(f"Original:\n{original_string}")

    find_regex = re.compile(input("Find Regex: "))
    print(f"Find:\n{find_regex}")
    replace_string = input("Replace String (use \\n for backrefs): ")
    print(f"Replace:\n{replace_string}")
    result = re.sub(find_regex, replace_string, original_string)
    print(f"Result:\n{result}")