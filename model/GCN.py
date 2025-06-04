#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/10 18:03
# @Author  : Wizard Chenhan Zhang
# @FileName: GCN.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv





class GCN(torch.nn.Module):
    def __init__(self, in_feature, out_class):
        super().__init__()

        self.with_relu = True
        self.with_bias = True

        self.hidden_sizes = [16]
        self.nfeat = in_feature
        self.nclass = out_class

        self.conv1 = GCNConv(in_feature, 16, cached=True,
                             normalize=True).half()
        self.conv2 = GCNConv(16, out_class, cached=True,
                             normalize=True).half()

        self.mseloss = torch.nn.MSELoss()
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training).half()
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
