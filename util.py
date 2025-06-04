import torch
import numpy as np
import pandas as pd
import networkx as nx
from utils import math_graph
from texttable import Texttable
from utils.math_utils import *
from utils.data_utils import *
import csv
from scipy.sparse.linalg import eigs
import random
import os


def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def table_reader(args):
    """
    Function to read the edges from the path.
    :param path: Path to the edge list.
    :return table: directed edges list
    """
    path = args.edges_path
    edges = pd.read_csv(path, header=None).values.tolist()
    return edges


def feature_reader(args):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: matrix of features.
    """
    file_path = '1'
    if args.dataset == 'BJ':
        file_path = '/home/xxx/Sy/data/beijing/rough_V.csv'
    elif args.dataset == 'SH':
        file_path = '/home/xxx/Sy/data/others/shanghai/fine_V.csv'
    elif args.dataset == 'D7':
        file_path = "/home/xxx/Sy/data/PemsD7/V_228.csv"
    elif args.dataset == 'D4':
        file_path = "/home/xxx/Sy/data/PemsD4/PemsD4_V.csv"
    elif args.dataset == 'LA':
        file_path = "/home/xxx/Projects/data/traffic/METR_LA/LA_imputed_V.csv"
    if file_path[-1] == "v":
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            data_seq = np.array([list(map(float, i)) for i in reader if i])
        features = data_seq.transpose()
    else:
        features = np.load(file_path).transpose((1, 2, 0)).transpose((0, 2, 1))
        features = features[:, :, 0]
    print('feature_size', features.shape) # (228, 12672)
    return features


def load_data(args, data_seq, n_frame=24):
    """
    Function to transform feature to dataSets
    :param args: the args from parser
    :return: The DataSets to train and test
    """
    print('data_seq', data_seq.shape)

    num_of_samples = data_seq.shape[0]
    print(num_of_samples)
    splitting_line1 = int(num_of_samples * args.train_ratio)
    splitting_line2 = int(num_of_samples * (args.test_ratio + args.train_ratio))

    seq_train = seq_gen(data_seq[: splitting_line1], n_frame)
    seq_val = seq_gen(data_seq[splitting_line1: splitting_line2], n_frame)
    seq_test = seq_gen(data_seq[splitting_line2:num_of_samples], n_frame)

    mean = np.mean(seq_train)
    std = np.std(seq_train)
    x_stats = {'mean': mean, 'std': std}

    x_train = z_score(seq_train, mean, std)
    x_val = z_score(seq_val, mean, std)
    x_test = z_score(seq_test, mean, std)

    train = x_train.transpose((0, 3, 1, 2))
    val = x_val.transpose((0, 3, 1, 2))
    test = x_test.transpose((0, 3, 1, 2))

    n_his = args.bash
    n_pre = args.pred

    train_x, train_y = train[:, :, : n_his, :], train[:, :, n_his:n_his + n_pre, :]
    val_x, val_y = val[:, :, : n_his, :], val[:, :, n_his:n_his + n_pre, :]
    test_x, test_y = test[:, :, : n_his, :], test[:, :, n_his:n_his + n_pre, :]

    if args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    train_x = train_x.transpose((0, 3, 2, 1)).squeeze(3)
    val_x = val_x.transpose((0, 3, 2, 1)).squeeze(3)
    test_x = test_x.transpose((0, 3, 2, 1)).squeeze(3)
    train_y = train_y.transpose((0, 3, 2, 1)).squeeze(3)
    val_y = val_y.transpose((0, 3, 2, 1)).squeeze(3)
    test_y = test_y.transpose((0, 3, 2, 1)).squeeze(3)

    train_x = torch.HalfTensor(train_x).to(device)
    train_y = torch.HalfTensor(train_y).to(device)
    val_x = torch.HalfTensor(val_x).to(device)
    val_y = torch.HalfTensor(val_y).to(device)
    test_x = torch.HalfTensor(test_x).to(device)
    test_y = torch.HalfTensor(test_y).to(device)

    # train_x = torch.FloatTensor(train_x).to(device)
    # train_y = torch.FloatTensor(train_y).to(device)
    # val_x = torch.FloatTensor(val_x).to(device)
    # val_y = torch.FloatTensor(val_y).to(device)
    # test_x = torch.FloatTensor(test_x).to(device)
    # test_y = torch.FloatTensor(test_y).to(device)

    x_data = {'train_x': train_x, 'train_y': train_y,
              'val_x': val_x, 'val_y': val_y,
              'test_x': test_x, 'test_y': test_y}
    dataset = Dataset(x_data, x_stats)
    return dataset


def edges2matrix_treated(args, edges, nodes):
    adj, adj_index = math_graph.edges2matrix(edges, nodes)
    length = adj.shape[0]
    W = math_graph.normalize_adj(adj + np.eye(adj.shape[0]))
    Lk = torch.from_numpy(W)
    Lk_index = torch.from_numpy(adj_index)
    for i in range(length):
        # print(i)
        for j in range(length):
            # print(j)
            # print('1', Lk_index[i][j][0])
            # print('2', Lk[i][j])
            Lk_index[i][j][0] = Lk[i][j]
    print(args.device)
    Lk = Lk.to(torch.float16)
    Lk = Lk.to(torch.float16).to(args.device)
    Lk_index = Lk_index.to(args.device)
    # print('LK_index', Lk_index.shape)

    return Lk, Lk_index


def matrix_reader(args):
    """
    Function to read the edges from the path.
    :param path: Path to the edge list.
    :return table: directed edges list
    """
    path = '1'
    if args.dataset == 'BJ':
        path = '/home/xxx/Sy/data/beijing/rough_A.csv'
    elif args.dataset == 'SH':
        path = '/home/xxx/Sy/data/others/shanghai/fine_A.csv'
    elif args.dataset == 'D7':
        path = "/home/xxx/Sy/data/PemsD7/W_228.csv"
    elif args.dataset == 'D4':
        path = "/home/xxx/Sy/data/PemsD4/PemsD4_W.csv"
    elif args.dataset == 'LA':
        path = "/home/xxx/Projects/data/traffic/METR_LA/LA_W.csv"

    print(path)
    if path[-1] == 'v':
        m = pd.read_csv(path, header=None).values
    else:
        m = np.load(path)
        m = m.astype(float)
    print(m.shape[0])
    return m


def matrix_reader2(args):
    """
    Function to read the edges from the path.
    :param path: Path to the edge list.
    :return table: directed edges list
    """
    path = '1'
    if args.dataset == 'BJ':
        path = '/home/xxx/Sy/data/beijing/rough_A.csv'
    elif args.dataset == 'SH':
        path = '/home/xxx/Sy/data/others/shanghai/fine_A.csv'
    elif args.dataset == 'D7':
        path = "/home/xxx/Sy/data/PemsD7/W_228.csv"
    elif args.dataset == 'D4':
        path = "/home/xxx/Sy/data/PemsD4/PemsD4_W.csv"
    elif args.dataset == 'LA':
        path = "/home/xxx/Projects/data/traffic/METR_LA/LA_W.csv"
    W = pd.read_csv(path, header=None).values
    Lk = math_graph.first_approx(W)
    Lk = torch.HalfTensor(Lk.astype(np.float16)).to(device=args.device)
    print('adj', Lk.shape[0])
    return Lk


def matrix_reader3(args):
    """
    Function to read and depose the adjacent matrix from the path.
    :param args: the args from parser
    :return: The Adjacent Matrix (treated)
    """
    path = '1'
    if args.dataset == 'BJ':
        path = '/home/xxx/Sy/data/beijing/rough_A.csv'
    elif args.dataset == 'SH':
        path = '/home/xxx/Sy/data/others/shanghai/fine_A.csv'
    elif args.dataset == 'D7':
        path =  "/home/xxx/Sy/data/PemsD7/W_228.csv"
    elif args.dataset == 'D4':
        path = "/home/xxx/Sy/data/PemsD4/PemsD4_W.csv"
    elif args.dataset == 'LA':
        path = "/home/xxx/Projects/data/traffic/METR_LA/LA_W.csv"
    # W = math_graph.load_matrix(path)
    W = math_graph.weight_matrix(path)

    Ks = 2
    L = math_graph.scaled_laplacian(W)
    Lk = math_graph.cheb_poly(L, Ks)
    # Lk = math_graph.first_approx(W)
    Lk = torch.HalfTensor(Lk.astype(np.float16))
    print('adj', Lk.shape[0])

    return Lk