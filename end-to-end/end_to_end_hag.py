import pickle
import torch
import torch.nn as nn
import torch_geometric.datasets
import torch.nn.functional as F
import time

from torch import Tensor



def Linear(input, in_channel, out_channel):
    weight = nn.Parameter(torch.Tensor(out_channel, in_channel)).to(device)
    return torch.matmul(input, weight.T)


def Aggr(x):
    # W = matrix.to_sparse()  # 转换为稀疏布尔型张量
    # return torch.sparse.mm(W, x)
    # print("x,matrix shape:",x.shape,matrix.shape)
    return torch.mm(matrix, x)


def qit_Aggr(x, flag):
    """
    list1存储的是顶点对，list2存放掩码格式的冗余串，长度是2708
    mode1:一般矩阵乘
    mode2:稀疏矩阵乘
    mode3:带冗余的一般矩阵乘
    mode0:带冗余的稀疏矩阵乘
    """


    if flag == 1:
        out1 = torch.mm(matrix, x)
        #print("原始计算时间:", time.time() - start_aggr)
        return out1

    if flag == 2:
        out1 = torch.mm(u_mat, x)
        out2 = torch.mm(r_mat, x)
        out=torch.mm(qit,out2)+out1
        #print("优化计算时间:", time.time() - start_aggr)
        return out
    if flag == 3:
        out4 = torch.sparse.mm(matrix_sp, x)
        #print("原始的稀疏乘计算时间:", time.time() - start_aggr)
        return out4

    if flag == 4:
        out5 = torch.sparse.mm(u_mat_sp, x)
        out6 = torch.sparse.mm(r_mat_sp, x)
        out = torch.sparse.mm(qit, out6) + out5
        #print("优化的稀疏乘计算时间:", time.time() - start_aggr)
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
    print("基于HAG聚合的GNN,数据集为：%s,设备为"%(name[id]),device)
    print("总用时：", time.time() - start_all)
    print("聚合阶段用时：",aggr_all_time)



dataset = ['cora', 'citeseer', 'pubmed', 'ppi', 'ogb']
device = torch.device('cpu')
classes = [7, 6, 3, 121, 41]
for id in range(4):
    name=dataset[id]
    if id<3:
        dataset = torch_geometric.datasets.Planetoid(root='../data/%s/' % name[id], name='%s' % name[id].capitalize())
    else:
        dataset=
    graph = dataset[0].to(device)
    in_channels = graph.x.shape[1]
    hidden_channels = 16
    out_channels = classes[id]
    matrix = torch.load('../matrix/%s_matrix.pt' % name[id]).float().to(device)
    u_mat = torch.load('../Rmat_Umat_QIT/HAG_%s_u_mat.pt' % name[id]).float().to(device)
    r_mat = torch.load('../Rmat_Umat_QIT/HAG_%s_r_mat.pt' % name[id]).float().to(device)
    qit = torch.load('../Rmat_Umat_QIT/HAG_%s_red_node.pt' % name[id]).float().T.to(device)

    u_mat_sp = Tensor.to_sparse(u_mat)
    r_mat_sp = Tensor.to_sparse(r_mat)
    qit_sp = Tensor.to_sparse(qit)
    matrix_sp = Tensor.to_sparse(matrix)
    aggr_all_time = 0





start = time.time()
inference()

