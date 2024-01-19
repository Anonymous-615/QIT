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
            # 原始聚合
            x = torch.sparse.mm(A_sparse, x)  # 矩阵乘法，这里的adjacency_matrix就是A矩阵

        if flag == 'qit':
            # QIT聚合
            if qit_max_flag:
                x = torch.sparse.mm(qit_r_mat_sp, x)
            else:
                x = torch.sparse.mm(qit_u_mat_sp, x)
        if flag == 'hag':
            # HAG聚合
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


# 假设数据集是一个特征矩阵X和邻接矩阵A
# 假设你有一个训练函数 train_model(model, data, labels)，用于模型训练
# 在这个函数中，你需要定义损失函数、优化器以及训练过程
# 这里提供一个简单的示例训练过程：

def train_model(model, data, labels, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = nn.NLLLoss()  # 负对数似然损失函数
    criterion = torch.nn.BCEWithLogitsLoss()        #给ppi用的
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


# 使用示例：
# 假设你有特征数据 X，标签数据 y，邻接矩阵 A
# 初始化模型
name = ['cora', 'citeseer', 'pubmed', 'ppi', 'ogb']
device = torch.device('cuda')
id = 3
# dataset = Planetoid(root='../data/%s'%name[id],name='Citeseer')
# X=dataset[0].x.to(device)
# y=dataset[0].y.to(device)
dataset = PPI(root='../data/PPI')
X = dataset[18].x.to(device)
y = dataset[18].y.to(device)

A = torch.load('D:\code\QIT_FINAL\matrix\%s_matrix.pt' % name[id]).float().to(device)
A_sparse = Tensor.to_sparse(A)
classes = [7, 6, 3, 121, 41]
qit_u_mat = torch.load('../rename/%s_u_mat.pt' % name[id]).float().to(device)
qit_r_mat = torch.load('../rename/%s_r_mat.pt' % name[id]).float().to(device)
# qit = torch.load('../rename/%s_red_node.pt' % name[id]).T.float().to(device)
qit = torch.load('../rename/%s_qit.pt' % name[id]).float().to(device)
qit_u_mat_sp = Tensor.to_sparse(qit_u_mat)
qit_r_mat_sp = Tensor.to_sparse(qit_r_mat)
qit_sp = Tensor.to_sparse(qit)
matrix_sp = Tensor.to_sparse(A)
hag_u_mat = torch.load('../rename/%s_HAG_u_mat.pt' % name[id]).float().to(device)
hag_r_mat = torch.load('../rename/%s_HAG_r_mat.pt' % name[id]).float().to(device)
# hag = torch.load('../rename/%s_HAG_red_node.pt' % name[id]).T.float().to(device)
hag = torch.load('../rename/%s_HAG_qit.pt' % name[id]).float().to(device)
hag_u_mat_sp = Tensor.to_sparse(hag_u_mat)
hag_r_mat_sp = Tensor.to_sparse(hag_r_mat)
hag_sp = Tensor.to_sparse(hag)

in_features = X.shape[1]  # 输入特征维度
hidden_features = 16  # 隐藏层特征维度

#out_features = len(torch.unique(y))  # 输出类别数，假设是分类任务
out_features=121    #给ppi用


mode = ['origin', 'qit', 'hag']
flag = mode[2]
gcn_model = GCN(in_features, hidden_features, out_features).to(device)
qit_max_flag = torch.sum(qit_r_mat) == max(torch.sum(qit_r_mat), torch.sum(qit_u_mat))

# 假设你有训练数据 data, labels
# 假设 data 是特征矩阵，labels 是对应的标签
# 这里还需要假设有一个邻接矩阵 adjacency_matrix，用于图结构表示
begin_time = time.time()
train_model(gcn_model, X, y)

str = f'数据集为：{name[id]}，设备为：{device}，聚合模式为：{flag}，训练时间为：{time.time() - begin_time}'
print(str)
