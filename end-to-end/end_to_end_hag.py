import pickle
import torch
import torch.nn as nn
import torch_geometric.datasets
import torch.nn.functional as F
import time

from torch import Tensor
from torch_geometric.datasets import PPI


def Linear(input, in_channel, out_channel):
    weight = nn.Parameter(torch.Tensor(out_channel, in_channel)).to(device)
    return torch.matmul(input, weight.T)


def Aggr(x):
    # W = matrix.to_sparse()  # 转换为稀疏布尔型张量
    # return torch.sparse.mm(W, x)
    # print("x,matrix shape:",x.shape,matrix.shape)
    return torch.mm(matrix, x)


def qit_Aggr(x, flag):



    if flag == 1:
        out1 = torch.mm(matrix, x)
        return out1

    if flag == 2:
        out1 = torch.mm(u_mat, x)
        out2 = torch.mm(r_mat, x)
        out=torch.mm(qit,out2)+out1
        return out
    if flag == 3:
        out4 = torch.sparse.mm(matrix_sp, x)
        return out4

    if flag == 4:
        out5 = torch.sparse.mm(u_mat_sp, x)
        out6 = torch.sparse.mm(r_mat_sp, x)
        out = torch.sparse.mm(qit, out6) + out5
        return out





def model(x):
    flag = 4
    global aggr_all_time
    x = Linear(x, in_channels, hidden_channels)

    start_aggr1 = time.time()
    x = qit_Aggr(x, flag)
    end_aggr1 = time.time()

    x = F.relu(x)
    x = Linear(x, hidden_channels, out_channels)

    start_aggr2 = time.time()
    x = qit_Aggr(x, flag)
    end_aggr2 = time.time()

    aggr_all_time += end_aggr2 - start_aggr2 + end_aggr1 - start_aggr1
    return F.log_softmax(x, dim=1)


def inference():
    start_all = time.time()
    for epoch in range(200):
        _, pred = model(graph.x).max(dim=1)
        # correct = float(pred[graph.test_mask].eq(graph.y[graph.test_mask]).sum().item())
        # acc = correct / graph.test_mask.sum().item()
    # print(f"acc : {acc}")
    print("HAG-aggr based inference, dataset: %s,device: "%(name[id]),device)
    print("Total time: ", time.time() - start_all)
    print("Aggr time:",aggr_all_time)



dataset = ['cora', 'citeseer', 'pubmed', 'ppi', 'ogb']
device = torch.device('cpu')
classes = [7, 6, 3, 121, 41]
for id in range(4):
    name=dataset[id]
    if id<3:
        dataset = torch_geometric.datasets.Planetoid(root='../data/%s/' % name[id], name='%s' % name[id].capitalize())
    else:
        dataset = PPI(root='../data/PPI')
    graph = dataset[0].to(device)
    in_channels = graph.x.shape[1]
    hidden_channels = 16
    out_channels = classes[id]
    matrix = torch.load('../data/original_adj_matrix/%s_matrix.pt' % name[id]).float().to(device)
    u_mat = torch.load('../data/split_adj_matrix/HAG_%s_u_mat.pt' % name[id]).float().to(device)
    r_mat = torch.load('../data/split_adj_matrix/HAG_%s_r_mat.pt' % name[id]).float().to(device)
    qit = torch.load('../data/split_adj_matrix/HAG_%s_red_node.pt' % name[id]).float().T.to(device)

    u_mat_sp = Tensor.to_sparse(u_mat)
    r_mat_sp = Tensor.to_sparse(r_mat)
    qit_sp = Tensor.to_sparse(qit)
    matrix_sp = Tensor.to_sparse(matrix)
    aggr_all_time = 0





start = time.time()
inference()

