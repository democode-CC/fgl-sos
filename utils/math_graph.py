import csv

import numpy as np

import pandas as pd

from scipy.sparse.linalg import eigs


import scipy.sparse as sp
import torch
from torch.nn.functional import normalize


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix

    Parameters
    ----------
    file_path: str, path of adjacency matrix file

    sigma2: float, default 0.1, scalar of matrix adj

    epsilon: float, default 0.5,
             thresholds to control the sparsity of matrix adj

    scaling: bool, default True, whether applies numerical scaling on adj

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, num_of_vertices)

    '''
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        adj = np.array([list(map(float, i)) for i in reader if i])

    # check whether adj is a 0/1 matrix.
    if set(np.unique(adj)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        adj = adj / 10
        mask = np.ones_like(adj) - np.identity(adj.shape[0])
        # refer to Eq.10
        exp = np.exp(- adj ** 2 / sigma2)
        return exp * (exp >= epsilon) * mask
    return adj

def first_approx(adj):
    '''
    1st-order approximation

    Parameters
    ----------
    adj: np.ndarray, adjacency matrix,
         shape is (num_of_vertices, num_of_vertices)

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, num_of_vertices)

    '''
    A = adj + np.identity(adj.shape[0])
    # sinvD = np.sqrt(np.matrix(np.diag(np.sum(A, axis=1))).I)
    sinvD = np.sqrt(np.linalg.pinv((np.diag(np.sum(A, axis=1)))))

    # refer to Eq.5
    # sinvD = np.array(sinvD)
    return np.identity(adj.shape[0]) + sinvD * A * sinvD


def load_matrix(file_path):
    return pd.read_csv(file_path, header=None).values.astype(float)


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)

def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])

    return np.asarray(LL)


def matrix2edges(matrix):
    """
    transform adjacent matrix to edges
    :param matrix: adjacent matrix
    :return: edges
    """
    edges = []
    for i in range(0, len(matrix)):
        edges.append([i, i])
        for j in range(0, len(matrix)):
            if matrix[i, j] == 1:
                edges.append([i, j])
    edge = np.array(edges)
    return edge


def edges2matrix(edges, nodes):
    """
    transform adjacent matrix to edges
    :param matrix: adjacent matrix
    :return: edges
    """
    length = nodes
    matrix = np.zeros([length, length])
    matrix_index = np.zeros([length, length, 3])
    for item in edges:
        if item[0] != item[1]:
            matrix_index[int(item[0])][int(item[1])] = [1, int(item[0]), int(item[1])]
            matrix[int(item[0])][int(item[1])] = 1
    # print('matrix', matrix)
    return matrix, matrix_index


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0)).to(matrix.device)
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian