import torch
import numpy as np
from my_parser import parameter_parser
from clustering import ClusteringMachine
from fed_trainer import Fed_Trainer
from util import load_data, feature_reader, matrix_reader, matrix_reader2, matrix_reader3, edges2matrix_treated
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import dense_to_sparse
import os
from my_parser import parameter_parser
from util import tab_printer, table_reader, set_seed
from trainer import Trainer
import time
import torch_geometric as pyg




def main():
    """
    Parsing command line parameters, reading data, train and test
    """
    time_start = time.time()
    args = parameter_parser()
    set_seed(args.seed)

    tab_printer(args)

    print(torch.randn(5))

    # Dataset preparation
    if args.dataset == 'cora':
        # if normalize
        dataset = pyg.datasets.Planetoid(root='./data/', name='Cora')
        data = dataset[0]
        # -> half
    elif args.dataset == 'citeseer':
        dataset = pyg.datasets.Planetoid(root='./data/', name='CiteSeer')
        data = dataset[0]
    elif args.dataset == 'pubmed':
        dataset = pyg.datasets.Planetoid(root='./data/', name='PubMed')
        data = dataset[0]
    

    
    elif args.dataset == 'polblogs':
        dataset = pyg.datasets.PolBlogs(root='./data/Polblogs/')
        data = dataset[0]



    adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    features = data.x
    labels = data.y

    # data: tensor -> numpy
    adj = adj.numpy()
    features = features.numpy()
    labels = labels.numpy()
    
    args.label_number = len(data.y.unique())




    # matrix = np.double(matrix_reader(args))
    # features = feature_reader(args)

    # Network partition
    cluster = ClusteringMachine(args, adj, features, labels)

    print('clusters have been obtained!')

    if not args.framework == 'Centralized':
        cluster.decompose()
        print('start training')
        trainer = Fed_Trainer(args, cluster, data)
        trainer.train()
        trainer.test()
    else:
        cluster.decompose()
        print('start training')
        trainer = Trainer(args, cluster, adj)
        trainer.train()
        trainer.test()

    torch.cuda.empty_cache()

    time_end = time.time()
    print('All done!')
    print('Total time: {:.2f} s'.format(time_end - time_start))

if __name__ == "__main__":
    main()

