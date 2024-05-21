# -*- coding:utf-8 -*-

import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid

import time
from torch_geometric.datasets import PPI

import torch.nn as nn
import torch.nn.functional as F
from gcnconv import GraphConvolution

from gatconv import GraphAttentionLayer
from graphsageconv import SAGEConv


class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(num_node_features, 32)
        self.gc2 = GraphConvolution(32, num_classes)

    def forward(self, data, adj):
        x = data.x
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, 32, dropout=0.5, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(32 * nheads, nclass, dropout=0.5, alpha=alpha, concat=False)

    def forward(self, data, adj):
        x = data.x
        x = F.dropout(x, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 32)
        self.conv2 = SAGEConv(32, num_classes)

    def forward(self, data, redun_free_edge_index):
        x, edge_index = data.x, redun_free_edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
