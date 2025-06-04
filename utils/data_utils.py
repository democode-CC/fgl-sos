import csv

import numpy as np


class Dataset(object):
    """
    Define the structure of DataSets
    """
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def __getitem__(self, key):
        return self.__data[key]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def z_inverse(self, type_):
        return self.__data[type_] * self.std + self.mean


def seq_gen(data_seq, n_frame):
    """
    Generate data in the form of standard sequence unit.
    :param data_seq: np.ndarray, time-series, shape is (length, num_of_vertices)
    :param n_frame: int, n_his + n_pred
    :return: np.ndarray, shape is (length - n_frame + 1, n_frame, num_of_vertices, 1)
    """
    data = np.zeros(shape=(data_seq.shape[0] - n_frame + 1,
                           n_frame, data_seq.shape[1], 1))
    for i in range(data_seq.shape[0] - n_frame + 1):
        data[i, :, :, 0] = data_seq[i: i + n_frame, :]
    return data
