import torch
from torch_geometric.datasets import Planetoid
from match import qit_match,hag_search_and_match,hag_pro_search_and_match,hag_pro_HUB_search_and_match


def edge2mat(edge_index, N):
    mat = torch.zeros([N,N])
    for i in range(edge_index[0].shape[0]):
        row,col=edge_index[0][i],edge_index[1][i]
        mat[row][col] = 1
    return mat


def mat2edge(mat):
    edge_index = torch.zeros([2,int(torch.sum(mat).item())],dtype=int)
    edge=torch.where(mat)
    for i in range(edge[0].shape[0]):
        row,col=int(edge[0][i]),int(edge[1][i])
        edge_index[0][i]=row
        edge_index[1][i]=col
    return edge_index


def redun_eliminate_qit(data):
    edge_index = data.edge_index
    N = data.x.shape[0]
    mat = edge2mat(edge_index, N)
    arst, qit, utt = qit_match(mat)
    #torch.save(mat2edge(utt),'temp_edge_index.pt')
    return mat2edge(utt)
    #return torch.load('./data/PPI/temp_edge_index.pt')
def redun_eliminate_hag(data):
    edge_index = data.edge_index
    N = data.x.shape[0]
    mat = edge2mat(edge_index, N)
    arst, qit, utt = hag_search_and_match(mat)
    #torch.save(mat2edge(utt), 'temp_edge_index_hag.pt')
    return mat2edge(utt)
    #return torch.load('temp_edge_index_hag.pt')
def redun_eliminate_hagPro(data):
    edge_index = data.edge_index
    N = data.x.shape[0]
    mat = edge2mat(edge_index, N)
    arst, qit, utt = hag_pro_search_and_match(mat)
    #torch.save(mat2edge(utt), 'temp_edge_index_hagP1.pt')
    return mat2edge(utt)
    #return torch.load('temp_edge_index_hagP1.pt')
def redun_eliminate_hagPro2(data):
    edge_index = data.edge_index
    N = data.x.shape[0]
    mat = edge2mat(edge_index, N)
    arst, qit, utt = hag_pro_HUB_search_and_match(mat)
    #torch.save(mat2edge(utt), 'temp_edge_index_hagP2.pt')
    return mat2edge(utt)
    # return torch.load('temp_edge_index_hagP2.pt')
# dataset = Planetoid(root='./data/cora', name='Cora')
# data = dataset[0]
# edge_index = data.edge_index
# print(edge_index.shape)
# print(torch.sum(edge2mat(edge_index,2708)))
# print(redun_eliminate(data).shape)