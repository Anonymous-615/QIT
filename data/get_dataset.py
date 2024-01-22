import torch
from torch_geometric.datasets import PPI, Planetoid


def edge_index_To_adj_matrix(edge,node_num):
    adj=torch.zeros(node_num,node_num)
    for i in range(edge.shape[1]):
        row,col=edge[0][i],edge[1][i]
        adj[row][col]=1
        adj[col][row]=1
    return adj

#cora
dataset0 = Planetoid(root='cora',name='Cora')
adj_mat0=edge_index_To_adj_matrix(dataset0[0].edge_index,dataset0[0].x.shape[0])
torch.save(adj_mat0,'cora_matrix.pt')

#citeseer
dataset1 = Planetoid(root='citeseer',name='Citeseer')
adj_mat1=edge_index_To_adj_matrix(dataset1[0].edge_index,dataset1[0].x.shape[0])
torch.save(adj_mat1,'citeseer_matrix.pt')

#pubmed
dataset2 = Planetoid(root='pubmed',name='Pubmed')
adj_mat2=edge_index_To_adj_matrix(dataset2[0].edge_index,dataset2[0].x.shape[0])
torch.save(adj_mat2,'pubmed_matrix.pt')

#PPI
dataset3 = PPI(root='PPI')
adj_mat3=edge_index_To_adj_matrix(dataset3[18].edge_index,dataset3[0].x.shape[0])
torch.save(adj_mat3,'ppi_matrix.pt')
