import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from utils.math_graph import *
from util import load_data, edges2matrix_treated
# from edmot import partition
import math
from matplotlib import pyplot as plt
import metis
from torch_geometric.data import Data
from torch_geometric import utils
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T
import pickle
import os

class ClusteringMachine(object):
    """
    Clustering the graph, feature set .
    """

    def __init__(self, args, edge, features, labels):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        """
        self.args = args
        self.edge = edge
        # self.graph = nx.from_edgelist(edge)
        self.graph = nx.from_numpy_array(edge)
        # if args.matrix_path[-1] == "v":
        #     with open(args.matrix_path, 'r') as f:
        #         reader = csv.reader(f)
        #         self.adj = np.array([list(map(float, i)) for i in reader if i])
        # else:
        #     self.adj = np.load(args.matrix_path)

        self.adj = edge
        self.features = features
        self.labels = labels
        self._set_sizes()
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = 12  # base windows
        self.class_count = 3  # predict window size, 1

    def decompose(self):
        """
        Decomposing the graph, partitioning the features, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        elif self.args.clustering_method == "random":
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        elif self.args.clustering_method == "load":
            print("\nload clustering started.\n")
            self.load_par()
        self.general_data_partitioning()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        # st = int 5993, parts = [number nodes, 1], maps each node to its cluster
        # Number of clusters, which is a hyperparameter, set to 10 here
        # Use dictionary to store each node's corresponding cluster
        self.clusters = list(set(parts))
        # Number of clusters, which is a hyperparameter, set to 10 here
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}
        # Use dictionary to store each node's corresponding cluster

    def load_par(self):
        parts = np.loadtxt('out/labels{:d}.txt'.format(self.args.cluster_number), dtype=int, delimiter=',')
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def save_clusters(self, save_dir='saved_clusters'):
        """
        Save the decomposed clusters to disk
        :param save_dir: Directory to save the clusters
        """
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir, self.args.dataset, str(self.args.cluster_number))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save cluster membership
        with open(os.path.join(save_dir, 'cluster_membership.pkl'), 'wb') as f:
            pickle.dump(self.cluster_membership, f)
            
        # Save subgraph data for each cluster
        for cluster in self.clusters:
            cluster_data = {
                'nodes': self.sg_nodes[cluster],
                'edges': self.sg_edges[cluster],
                'features': self.sg_features[cluster],
                'labels': self.sg_labels[cluster],
                'matrix': self.sg_matrix[cluster],
                'edge_index': self.sg_edge_index[cluster],
                'adj_t': self.sg_adj_t[cluster],
                'data': self.sg_data[cluster]
            }
            with open(os.path.join(save_dir, f'cluster_{cluster}.pkl'), 'wb') as f:
                pickle.dump(cluster_data, f)
                
        print(f"Clusters saved to {save_dir}")

    def load_clusters(self, load_dir='saved_clusters'):
        """
        Load the decomposed clusters from disk
        :param load_dir: Directory containing the saved clusters
        """
        # Load cluster membership
        load_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), load_dir, self.args.dataset, self.args.cluster_number)
        with open(os.path.join(load_dir, 'cluster_membership.pkl'), 'rb') as f:
            self.cluster_membership = pickle.load(f)
            
        self.clusters = list(set(self.cluster_membership.values()))
        
        # Load subgraph data for each cluster
        for cluster in self.clusters:
            with open(os.path.join(load_dir, f'cluster_{cluster}.pkl'), 'rb') as f:
                cluster_data = pickle.load(f)
                self.sg_nodes[cluster] = cluster_data['nodes']
                self.sg_edges[cluster] = cluster_data['edges']
                self.sg_features[cluster] = cluster_data['features']
                self.sg_labels[cluster] = cluster_data['labels']
                self.sg_matrix[cluster] = cluster_data['matrix']
                self.sg_edge_index[cluster] = cluster_data['edge_index']
                self.sg_adj_t[cluster] = cluster_data['adj_t']
                self.sg_data[cluster] = cluster_data['data']
                
        print(f"Clusters loaded from {load_dir}")

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        print('general')
        self.sg_nodes = {}
        # Dictionary, length = 10, stores nodes for each cluster
        self.sg_edges = {}
        # Dictionary, length = 10, stores edges for each cluster
        self.sg_features = {}
        # Dictionary, length = 10, stores node features for each cluster, corresponding to sg_nodes
        self.sg_labels = {}
        # Dictionary, length = 10, stores node labels for each cluster, corresponding to sg_nodes
        self.sg_matrix = {}
        self.sg_edge_index = {}
        self.sg_data = {}
        self.sg_adj_t = {}


        for cluster in self.clusters:
            subgraph = self.graph.subgraph(
                [node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [
                [mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster], :]
            self.sg_labels[cluster] = self.labels[self.sg_nodes[cluster]].transpose()
            self.sg_matrix[cluster], _ = edges2matrix_treated(self.args, self.sg_edges[cluster], len(self.sg_nodes[cluster]))
            self.sg_edge_index[cluster] = utils.dense_to_sparse(self.sg_matrix[cluster])[0]
            self.sg_adj_t[cluster] = utils.dense_to_sparse(self.sg_matrix[cluster])[0]

            if self.args.model == 'GCN':
                self.sg_data[cluster] = Data(x=torch.HalfTensor(self.sg_features[cluster]).to(self.args.device),
                                             edge_index=torch.tensor(self.sg_edge_index[cluster]).to(self.args.device),
                                             y=torch.LongTensor(self.sg_labels[cluster]).to(self.args.device))
            else:
                self.sg_data[cluster] = Data(x=torch.HalfTensor(self.sg_features[cluster]).to(self.args.device),
                                             edge_index=torch.tensor(self.sg_edge_index[cluster]).to(self.args.device),
                                             y=torch.LongTensor(self.sg_labels[cluster]).to(self.args.device))





            total_num = self.sg_features[cluster].shape[0]
            train_num = int(self.sg_features[cluster].shape[0] * 0.8)
            val_num = int(self.sg_features[cluster].shape[0] * 0.1)
            test_num = int(self.sg_features[cluster].shape[0] * 0.1)

            # Train/val/test split
            self.sg_data[cluster].train_mask = torch.zeros((total_num, ), dtype=torch.bool).to(self.args.device)
            self.sg_data[cluster].train_mask[0:train_num] = True

            self.sg_data[cluster].val_mask = torch.zeros((total_num, ), dtype=torch.bool).to(self.args.device)
            self.sg_data[cluster].val_mask[train_num:train_num+val_num] = True

            self.sg_data[cluster].test_mask = torch.zeros((total_num, ), dtype=torch.bool).to(self.args.device)
            self.sg_data[cluster].test_mask[train_num+val_num:train_num+val_num+test_num] = True

            print('self.sg_data[cluster]', self.sg_data[cluster])

        # Save the decomposed clusters
        self.save_clusters()








