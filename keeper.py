import FLAGS
import os
import Utilities
import skimage.io as io
from scipy.spatial import distance
import time as t
import datetime
import numpy as np
import operator


def store_tensor_list(arg_list, func_to_call, log_filename):
    with open(log_filename, "w") as f:
        for arg in arg_list:
            tensor_list = func_to_call(arg)
            f.write(arg + " " + str(len(tensor_list)) + '\n')
            for tensor in tensor_list:
                [f.write(str(s) + ' ') for s in tensor.shape]
                f.write('\n')

                [f.write(str(v) + ' ') for v in list(np.array(tensor).reshape(-1))]
                f.write('\n')


def get_tensor_list_dict_from_disk(log_filename):
    start = t.time()
    print('Retrieving tensors from', log_filename, "...")

    arg_list_dict = {}
    with open(log_filename, 'r') as f:
        lines = f.readlines()
        walker = 0
        while walker < len(lines):
            arg_filename, n_blocks = lines[walker].rstrip().split(' ')
            walker += 1
            tensors_list = []
            for _ in range(int(n_blocks)):
                shape = [int(v) for v in lines[walker].rstrip().split(' ')]
                walker += 1

                tensor = np.array([int(v) for v in lines[walker].rstrip().split(' ')]).reshape(shape)
                walker += 1

                tensors_list.append(tensor)

            arg_list_dict[arg_filename] = tensors_list

    finish = t.time()
    print("Finished retrieving tensors in", datetime.timedelta(seconds=finish-start))
    return arg_list_dict

