__author__ = 'Juxtux'
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def file_to_matrix(file_x):
    r_list = list()

    while 1:
        rows = file_x.readline()
        if not rows or rows == '\n':
            break
        rows = rows.strip().split('\t')
        row = [float(num) for num in rows if is_number(num)]
        r_list.append(row)

    r_array = np.array(r_list)
    r_array = np.transpose(r_array)
    return np.delete(r_array,0,1)

def file_to_y(file_y):
    r_list = []
    while 1:
        row = file_y.readline()
        if not row:
            break
        if is_number(row[0:2]):
            row_s = row.split('\t\t')
            result = row_s[1].strip()
            r_list.append([int(row[0:2]), result])
    return r_list

def y_array(y, zeros, ones):
    r_array = [-1 if tuple[1] == zeros else 1 if tuple[1] == ones else 'error_value' for tuple in y]
    return np.array(r_array)