#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/10 18:03
# @Author  : Wizard Chenhan Zhang
# @FileName: GCN.py
# @Software: PyCharm
import copy

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse





class GCN_IB(torch.nn.Module):
    def __init__(self, in_feature, out_class):
        super(GCN_IB, self).__init__()

        self.with_relu = True
        self.with_bias = True

        self.hidden_sizes = [16]
        self.nfeat = in_feature
        self.nclass = out_class


        self.conv1 = GCNConv(in_feature, 16, cached=True,
                             normalize=True).half()
        self.conv2 = GCNConv(16, out_class, cached=True,
                             normalize=True).half()



        first_dense_neurons = 16
        second_dense_neurons = 2    # assignment

        self.fully_connected_1 = torch.nn.Linear(out_class,
                                                 first_dense_neurons)
        self.fully_connected_2 = torch.nn.Linear(first_dense_neurons,
                                                 second_dense_neurons)
        self.mseloss = torch.nn.MSELoss()
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def assignment2adj(self, assignment, adj):
        adj = copy.deepcopy(adj)
        for i in range(assignment.shape[0]):
            if assignment[i][0] >= assignment[i][1]:
                pass
            else:
                adj[i] = torch.zeros(assignment.shape[0])
                adj[:][i] = 0
        return adj

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # For IB
        node_features_1 = F.relu(self.conv1(x, edge_index, edge_weight)).half()
        node_features_2 = self.conv2(node_features_1, edge_index, edge_weight).half()


        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))  # torch.Size([696, 16])
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)  # torch.Size([696, 2])

        # group_features = S^T x X_l
        group_features = torch.mm(torch.t(assignment),node_features_2)  # torch.Size([2, 7])
        # local_features = torch.mm()

        EYE_global = torch.ones(2).half().to(x.device)
        Eye_local = torch.ones(x.shape[0]).half().to(x.device)
        adj = to_dense_adj(edge_index)[0].half()
        adj.requires_grad = False

        # Global new adj representation
        # new_adj = S^T x A x S
        new_adj_global = torch.mm(torch.t(assignment),adj)  # Assignment is S
        new_adj_global = torch.mm(new_adj_global,assignment)

        new_adj = self.assignment2adj(assignment, adj)
        new_edge_index = dense_to_sparse(new_adj)[0]

        positive = torch.clamp(group_features[0].unsqueeze(dim=0),-100,100)
        negative = torch.clamp(group_features[1].unsqueeze(dim=0),-100,100)

        normalize_new_adj = F.normalize(new_adj_global, p=1, dim=1)
        norm_diag = torch.diag(normalize_new_adj)
        pos_penalty = self.mseloss(norm_diag, EYE_global)


        # Local new adj representation
        # local_pos_penalty = self.mseloss(Eye_local)










        # node_features_2  torch.Size([696, 7]) - > node embedding
        # assignment torch.Size([696, 2])
        graph_embedding = torch.mm(torch.t(assignment), node_features_2)
        graph_embedding = torch.mean(graph_embedding, dim=0, keepdim=True)


        # For Classification



        x = F.relu(self.conv1(x, new_edge_index, edge_weight)).half()
        x = F.dropout(x, training=self.training).half()
        x = self.conv2(x, new_edge_index, edge_weight)

        node_embedding = node_features_2

        return new_edge_index, node_embedding, graph_embedding, positive, negative,  pos_penalty, F.log_softmax(x, dim=1).half() # return 5 outputs



class Local_Discriminator(torch.nn.Module): # A fully connected network
    def __init__(self, args):
        super(Local_Discriminator, self).__init__()
        self.args = args
        second_gcn_dimensions = self.args.label_number
        dis_hidden_dimensions = 4

        self.input_size = 2 * second_gcn_dimensions
        self.hidden_size = dis_hidden_dimensions
        self.fc1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

        torch.nn.init.constant(self.fc1.weight, 0.01)
        torch.nn.init.constant(self.fc2.weight, 0.01)

    def forward(self, embeddings, positive): # embedding is node embedding
        positive = positive.expand(embeddings.shape[0], positive.shape[1])
        cat_embeddings = torch.cat((embeddings, positive), dim=-1)
        pre = self.relu(self.fc1(cat_embeddings))  # shape: torch.Size([1694, 12])
        pre = self.fc2(pre)
        return pre

class Global_Discriminator(torch.nn.Module): # A fully connected network
    def __init__(self, args):
        super(Global_Discriminator, self).__init__()
        self.args = args
        second_gcn_dimensions = self.args.label_number
        dis_hidden_dimensions = 4
        self.input_size = 2 * second_gcn_dimensions
        self.hidden_size = dis_hidden_dimensions
        self.fc1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

        torch.nn.init.constant(self.fc1.weight, 0.01)
        torch.nn.init.constant(self.fc2.weight, 0.01)

    def forward(self, embeddings, positive):
        cat_embeddings = torch.cat((embeddings, positive), dim=-1)
        pre = self.relu(self.fc1(cat_embeddings))
        pre = self.fc2(pre)
        return pre