import csv

def get_data(csv_file):
    with open(csv_file, 'r') as csvfile:
        lines  = [line for line in csv.reader(csvfile, delimiter=',')]
        csvfile.close()

    data = {}
    features = lines[0]
    for feature in features:
        data[feature] = []
    lines.pop(0)
    for line in lines:
        for i, value in enumerate(line):
            data[features[i]].append(value)
    return data

def is_float(string):
    if isinstance(string, float) or isinstance(string, int):
        return True
    if len(string) == 0:
        return False
    if len(string) == 1:
        if not string[0] in "0123456789":
            return False
        else:
            return True
    if len(string) == 2 and (string == "-." or string == ".-"):
        return False
    if string.count('.') > 1:
        return False
    if '-' in string[1:]:
        return False
    for c in string:
        if not (c == '-' or c == '.' or c in "0123456789"):
            return False
    return True

def is_list_num(lst):
    for elem in lst:
        if elem != "" and not is_float(elem):
            return False
    return True

def quicksort(lst, rev=False):
    if len(lst) < 2:
        return lst
    else:
        pivot = lst[0] if rev == False else lst[-1]
        less = [i for i in lst[1:] if i <= pivot] if rev == False\
                            else [i for i in lst[:-1] if i <= pivot]
        greater = [i for i in lst[1:] if i > pivot] if rev == False\
                            else [i for i in lst[:-1] if i > pivot]
        return quicksort(less, rev=rev) + [pivot] + quicksort(greater, rev=rev) if rev == False\
                else quicksort(greater, rev=rev) + [pivot] + quicksort(less, rev=rev)

def keep_only_float(lst):
    ret = []
    for elem in lst:
        if is_float(elem):
            ret.append(float(elem))
    return ret

def is_int(f):
    return int(f) == f
