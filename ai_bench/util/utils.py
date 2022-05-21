from collections import Iterable


def to_dict(obj):
    data = {}
    if isinstance(obj, Iterable):
        data = []
        for temp_obj in obj:
            temp_data = {}
            for key, value in temp_obj.__dict__.items():
                try:
                    temp_data[key] = to_dict(value)
                except AttributeError:
                    temp_data[key] = value
            data.append(temp_data)
    else:
        for key, value in obj.__dict__.items():
            try:
                data[key] = to_dict(value)
            except AttributeError:
               data[key] = value
    return data


def binary_search(A, val):
    low, high = 0, len(A) - 1
    if A[high] < val:
        return high
    while low < high:
        mid = low + (high - low) // 2
        if A[mid] >= val:
            high = mid
        else:
            low = mid + 1
    return low
