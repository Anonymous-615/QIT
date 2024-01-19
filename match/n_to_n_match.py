import torch
from hiper_heuristic_sort import get_sorted
from get_neighbor import get_neighbor_for_redun


def loop_of_match(adjmat, redundancy_mask, redundancy_node):
    base = torch.zeros(size=[adjmat.shape[0]])
    hag_index = get_sorted(redundancy_mask, redundancy_node, 'qit')
    boost = 0
    remain = torch.clone(adjmat)
    redun_mat_list = []
    redun_node_list = []

    for i in range(hag_index.shape[0]):
        ide = hag_index[i]
        node = torch.where(redundancy_node[ide] != 0)[0]
        redun_mask = redundancy_mask[hag_index[i]]
        index = torch.where(redun_mask != 0)[0]
        is_subset = remain[node][:, index]
        available_node_num = torch.sum(torch.all(is_subset, dim=1))

        if available_node_num > 1:
            available_node = node[torch.all(is_subset, dim=1)]
            boost += torch.sum(redun_mask) * available_node_num
            remain[available_node] -= redun_mask
            redun_mat_list.append(redun_mask)
            node_mask = torch.clone(base)
            node_mask[available_node] = 1
            redun_node_list.append(node_mask)
    redun_mat = torch.tensor([aa.tolist() for aa in redun_mat_list])
    redun_node = torch.tensor([aa.tolist() for aa in redun_node_list])
    # print("节省的操作数：", boost)
    # print("剩余的操作数：", torch.sum(remain))
    torch.save(redun_mat, '../data/split-adj-matrix/%s_r_mat.pt'%name)
    torch.save(redun_node.T, '../data/split-adj-matrix/%s_qit.pt'%name)
    torch.save(remain,'../data/split-adj-matrix/%s_u_mat.pt'%name)
    #return redun_mat, redun_node, boost


def nvn():
    adj_matrix = torch.load("../data/origin-adj-matrix/%s_matrix.pt" % name).float().to(device)
    red_mask = torch.load('../search/output/%s_redun_mask.pt' % name).to(device)
    red_node = get_neighbor_for_redun(red_mask, adj_matrix)
    #available_red_mat1, available_red_node1, boost =
    loop_of_match(adj_matrix, red_mask, red_node)
    #print("计算冗余串需要的操作数：", torch.sum(available_red_mat1))
    # print("净提升：", boost - torch.sum(available_red_mat1))
    # print()



dataset=['cora','citeseer','ppi','pubmed']
device = torch.device('cuda')
for id in range(4):
    name = dataset[id]
    nvn()
    output=f'dataset:{name}, many-to-many match complete'
    print(output)
