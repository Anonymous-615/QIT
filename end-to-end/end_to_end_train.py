import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
import pickle
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = Linear(in_features, out_features)

    def forward(self, x):
        global flag, qit_max_flag
        x = self.linear(x)

        if flag == 'origin':
            # origin aggr
            x = torch.sparse.mm(A_sparse, x)

        if flag == 'qit':
            # QIT aggr
            if qit_max_flag:
                x = torch.sparse.mm(qit_r_mat_sp, x)
            else:
                x = torch.sparse.mm(qit_u_mat_sp, x)
        if flag == 'hag':
            # HAG aggr
            out5 = torch.sparse.mm(hag_u_mat_sp, x)
            out6 = torch.sparse.mm(hag_r_mat_sp, x)
            x = torch.sparse.mm(hag_sp, out6) + out5
        return x


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)




def train_model(model, data, labels, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = nn.NLLLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        #print("debug:",output.shape,labels.shape)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")



name = ['cora', 'citeseer', 'pubmed', 'ppi', 'ogb']
# classes = [7, 6, 3, 121, 41]
device = torch.device('cuda')
id = 3


if name[id]=='cora':
    dataset = Planetoid(root='../data/%s'%name[id],name='Cora')
    X=dataset[0].x.to(device)
    y=dataset[0].y.to(device)
    out_features = len(torch.unique(y))
if name[id]=='citeseer':
    dataset = Planetoid(root='../data/%s'%name[id],name='Citeseer')
    X=dataset[0].x.to(device)
    y=dataset[0].y.to(device)
    out_features = len(torch.unique(y))
if name[id] == 'pubmed':
    dataset = Planetoid(root='../data/%s'%name[id],name='Pubmed')
    X=dataset[0].x.to(device)
    y=dataset[0].y.to(device)
    out_features = len(torch.unique(y))


if name[id] == 'ppi':
    dataset = PPI(root='../data/PPI')
    X = dataset[18].x.to(device)
    y = dataset[18].y.to(device)
    out_features = 121

A = torch.load('../data/original_adj_matrix/%s_matrix.pt' % name[id]).float().to(device)
A_sparse = Tensor.to_sparse(A)

qit_u_mat = torch.load('../data/split_adj_matrix/%s_u_mat.pt' % name[id]).float().to(device)
qit_r_mat = torch.load('../data/split_adj_matrix/%s_r_mat.pt' % name[id]).float().to(device)
# qit = torch.load('../rename/%s_red_node.pt' % name[id]).T.float().to(device)
qit = torch.load('../data/split_adj_matrix/%s_qit.pt' % name[id]).float().to(device)
qit_u_mat_sp = Tensor.to_sparse(qit_u_mat)
qit_r_mat_sp = Tensor.to_sparse(qit_r_mat)
qit_sp = Tensor.to_sparse(qit)
matrix_sp = Tensor.to_sparse(A)
hag_u_mat = torch.load('../data/split_adj_matrix/%s_HAG_u_mat.pt' % name[id]).float().to(device)
hag_r_mat = torch.load('../data/split_adj_matrix/%s_HAG_r_mat.pt' % name[id]).float().to(device)
# hag = torch.load('../rename/%s_HAG_red_node.pt' % name[id]).T.float().to(device)
hag = torch.load('../data/split_adj_matrix/%s_HAG_qit.pt' % name[id]).float().to(device)
hag_u_mat_sp = Tensor.to_sparse(hag_u_mat)
hag_r_mat_sp = Tensor.to_sparse(hag_r_mat)
hag_sp = Tensor.to_sparse(hag)

in_features = X.shape[1]
hidden_features = 16

mode = ['origin', 'qit', 'hag']
flag = mode[2]
gcn_model = GCN(in_features, hidden_features, out_features).to(device)
qit_max_flag = torch.sum(qit_r_mat) == max(torch.sum(qit_r_mat), torch.sum(qit_u_mat))


begin_time = time.time()
train_model(gcn_model, X, y)

str = f'dataset: {name[id]}, device: {device}, aggregation mode: {flag}，train time: {time.time() - begin_time}'
print(str)
