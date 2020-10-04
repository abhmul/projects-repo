def range2(start, end=None, step=1):
    """1-index inclusive range"""
    if end is None:
        return range(1, start + 1, step)
    else:
        return range(start, end + 1, step)
