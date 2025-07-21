from LQO.constant import *

def swap_dict_items(data, key1, key2):
    if key1 not in data or key2 not in data:
        raise KeyError("Index Error")
    items = list(data.items())
    index1 = next(i for i, (k, v) in enumerate(items) if k == key1)
    index2 = next(i for i, (k, v) in enumerate(items) if k == key2)
    items[index1], items[index2] = items[index2], items[index1]
    new_data = dict(items)
    return new_data

def minmax_transform(x, min_max):
    min_val, max_val = min_max
    if min_val == max_val:
        return 0.0
    return (x - min_val) / (max_val - min_val)


def planByGeqo():
    hints = [
        "SET GEQO TO ON;",
        "SET geqo_threshold TO 2;"
    ]
    return hints