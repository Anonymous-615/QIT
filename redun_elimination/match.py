import random
import time
import torch


def gemm_based_search(mat):
    redun = torch.mm(mat, mat.T).fill_diagonal_(0)
    redun[redun < 2] = 0
    indice = torch.where(redun != 0)
    red_mask = mat[indice[0]] * mat[indice[1]]
    unique = torch.unique(red_mask, dim=0, return_inverse=True)[0]
    return unique


def multi_level_search(mat):
    redun0 = gemm_based_search(mat)
    redun1 = gemm_based_search(redun0)
    redun2 = gemm_based_search(redun1)
    redun3 = gemm_based_search(redun2)
    unique = torch.unique(torch.cat([redun0, redun1, redun2, redun3]), dim=0, return_inverse=True)[0]
    return unique


def get_neighbors(redun, mat):
    neighbors = torch.zeros([redun.shape[0], mat.shape[0]])
    for i in range(redun.shape[0]):
        inter = mat * redun[i]
        neighbors_index = torch.nonzero(
            torch.all(torch.eq(inter, redun[i].unsqueeze(0).expand_as(inter)), dim=1)).flatten().tolist()
        neighbors[i][neighbors_index] = 1
    return neighbors


def get_adj(redun, neighbors):
    redun_adj = torch.ones([redun.shape[0], redun.shape[0]])
    redun_elements = torch.mm(redun, redun.T).fill_diagonal_(0)
    for i in range(redun_elements.shape[0]):
        inter_index = torch.where(redun_elements[i])[0]
        for j in range(inter_index.shape[0]):
            if torch.sum(neighbors[i] * neighbors[inter_index[j]]):
                redun_adj[i][inter_index[j]] = 0
    return redun_adj


def generate_three_floats():
    float1 = random.uniform(0, 1)
    float2 = random.uniform(0, 1 - float1)
    float3 = 1 - float1 - float2
    return float1, float2, float3


def get_sorted(mask, neighbor, adj):
    # name = 'conflict'
    # name = 'hag'
    # name = 'hag-pro'
    # name = 'order'
    name = 'rand'
    redun_len = torch.sum(mask, dim=1)
    redun_node_len = torch.sum(neighbor, dim=1)
    redun_conflict = torch.sum(adj, dim=1)
    if name == 'flip':
        return torch.flip(torch.arange(torch.sum(mask, dim=1).shape[0]), dims=[0])
    if name == 'order':
        return torch.arange(torch.sum(mask, dim=1).shape[0])
    if name == 'rand':
        original_sequence = torch.arange(torch.sum(mask, dim=1).shape[0])
        shuffled_indices = torch.randperm(original_sequence.size(0))
        shuffled_sequence = original_sequence[shuffled_indices]
        return shuffled_sequence, [1, 1, 1]
    if name == 'hag':
        return torch.sort(redun_len, descending=True)[1], [1, 1, 1]  # HAG
    if name == 'hag-pro':
        return torch.sort(redun_node_len, descending=True)[1]  # HAG
    if name == 'qit':
        return torch.sort(redun_len * redun_node_len, descending=True)[1]  # QIT
    if name == 'conflict':
        a, b, c = generate_three_floats()
        arg = [a, b, c]
        # print(a,b,c)
        random.shuffle(arg)
        return torch.sort(arg[0] * redun_len + arg[1] * redun_node_len + arg[2] * redun_conflict, descending=True)[
            1], arg


def loop_of_match(adjmat):
    redundancy_mask = gemm_based_search(adjmat)
    #torch.save(redundancy_mask, 'pubmed_rst.pt')
    #redundancy_mask =torch.load('./data/pubmed/pubmed_rst.pt')
    #print("rst processing complete")
    redundancy_node = get_neighbors(redundancy_mask, adjmat)
    #torch.save(redundancy_node, 'pubmed_neighbor.pt')
    #redundancy_node = torch.load('./data/pubmed/pubmed_neighbor.pt')
    #print("neighbor processing complete")
    redun_adj = get_adj(redundancy_mask, redundancy_node)
    #torch.save(redun_adj, 'pubmed_redun_adj.pt')
    #redun_adj = torch.load('./data/pubmed/pubmed_redun_adj.pt')
    #print("redun adj processing complete")
    hag_index, arg = get_sorted(redundancy_mask, redundancy_node, redun_adj)
    #hag_index, arg =torch.randperm(redundancy_mask.size(0)) ,[1,1,1]
    base = torch.zeros(size=[adjmat.shape[0]])
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
    return redun_mat, redun_node.T, remain, boost, arg


def qit_match(mat):
    maxi = 0
    for i in range(1):
        #print("epoch:",i)
        arst, qit, utt, temp, arg = loop_of_match(mat)
        if temp > maxi:
            maxi = temp
            m_arst,m_qit,m_utt=arst, qit, utt
    return m_arst,m_qit,m_utt


def hag_search_and_match(mat):

    covered_node = torch.zeros([mat.shape[0]])
    redun = torch.mm(mat, mat.T).fill_diagonal_(0)
    redun[redun < 2] = 0
    indice = torch.where(redun != 0)
    red_mask = mat[indice[0]] * mat[indice[1]]


    
    neighbors = torch.zeros([red_mask.shape[0], mat.shape[0]])
    for i in range(neighbors.shape[0]):
        row,col=indice[0][i],indice[1][i]
        neighbors[i][row] = 1
        neighbors[i][col] = 1
        
    # index = torch.randperm(unique.size(0))
    index = torch.sort(torch.sum(red_mask, dim=1), descending=True)[1]
    rst, neighbor = red_mask[index], neighbors[index]
    arst_ind = []


    for i in range(mat.shape[0]//4):
        row, col = torch.where(neighbor[i])[0][0], torch.where(neighbor[i])[0][1]
        if covered_node[row] == 0 and covered_node[col] == 0:
            arst_ind.append(i)
            covered_node[row] = 1
            covered_node[col] = 1
    arst = rst[arst_ind]
    qit = neighbor[arst_ind].T
    utt = mat - torch.mm(qit, arst)
    return arst, qit, utt


def hag_pro_search_and_match(mat):

    k = mat.shape[0]//2
    rst = torch.zeros([k, mat.shape[0]])
    neighbor = torch.zeros_like(rst)
    base = torch.zeros([mat.shape[0]])
    # index = torch.sort(torch.sum(mat, dim=1), descending=True)[1]
    index = torch.randperm(mat.size(0))
    for i in range(k):
        temp1 = torch.clone(base)
        temp1[index[2 * i]] = 1
        temp1[index[2 * i + 1]] = 1
        rst[i] = temp1
        available_nodes_index = torch.where(mat[index[2 * i]] * mat[index[2 * i + 1]])[0]
        neighbor[i][available_nodes_index] = 1
    arst = rst
    qit = neighbor.T
    utt = mat - torch.mm(qit, arst)
    return arst, qit, utt


def hag_pro_HUB_search_and_match(mat):

    k = mat.shape[0]
    neighbor = torch.zeros([k, mat.shape[0]])
    base = torch.zeros([mat.shape[0]])
    index = torch.sort(torch.sum(mat, dim=1), descending=True)[1]
    #index = torch.randperm(mat.size(0))
    covered_nodes=torch.zeros([k])
    arst_list=[]
    for i in range(k):
        temp1 = torch.clone(base)
        temp1[index[i]] = 1
        neighbors=torch.where(mat[index[i]])[0]
        if neighbors.shape[0]:
            index2 = torch.sort(torch.sum(mat[neighbors], dim=1), descending=True)[1]
            #print(index2.shape,neighbors[index2[0]].shape)
            temp1[neighbors[index2[0]]] = 1
        uncovered_nodes=torch.where(covered_nodes==0)[0]
        for j in range(uncovered_nodes.shape[0]):
            if torch.equal(mat[uncovered_nodes[j]]*temp1,temp1):
                arst_list.append(temp1)
                covered_nodes[uncovered_nodes[j]]=1
                neighbor[len(arst_list)-1][uncovered_nodes[j]]=1
    arst = torch.stack(arst_list)
    qit = neighbor[0:arst.shape[0]].T
    utt = mat - torch.mm(qit, arst)
    return arst, qit, utt

