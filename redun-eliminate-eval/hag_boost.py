import torch

dataset=['cora','citeseer','ppi','pubmed']
device = torch.device('cuda')
for id in range(4):
    name = dataset[id]
    adj_matrix = torch.load("../data/origin-adj-matrix/%s_matrix.pt" % name).float().to(device)
    redun_mat=torch.load( '../data/split-adj-matrix/%s_HAG_r_mat.pt'%name)
    redun_node_T=torch.load( '../data/split-adj-matrix/%s_HAG_qit.pt'%name)
    remain=torch.load('../data/split-adj-matrix/%s_HAG_u_mat.pt'%name)
    output=(f'dataset: {name}, Aggregation speed-up based on HAG redun elimination: {torch.sum(remain)/torch.sum(adj_matrix)}, available redun set cost: {torch.sum(redun_mat)}, data dependency set size: {torch.sum(redun_node_T)}, '
            f'redun elimination scale: {torch.sum(adj_matrix)-torch.sum(remain)-torch.sum(redun_mat)}')
    print(output)