import torch

torch.set_printoptions(profile="full")
device = torch.device('cuda')


def get_sorted(mask):
    redun_len = torch.sum(mask, dim=1)
    return torch.sort(redun_len, descending=True)[1]  # HAG


def get_neighbor_for_redun(red_mask, mat):
    """
    找到冗余集中每个节点对给定邻接矩阵的所有对应关系
    :return:
    red_mask:图中所有的冗余串的掩码，size=[4,2708]
    mat:size=[131,2708]
    red_node:[4,131]
    """
    base = torch.zeros(size=[mat.shape[0]])
    red_node = torch.zeros(size=[red_mask.shape[0], mat.shape[0]], dtype=torch.long)

    for i in range(red_mask.shape[0]):
        red = red_mask[i]
        red_to_mat = torch.mul(mat, red)
        sub_set_index = torch.all(red_to_mat == red, dim=1)
        #print(sub_set_index)
        temp_mask = torch.clone(base)
        if torch.where(sub_set_index)[0].shape[0]>1:
            temp_mask[torch.where(sub_set_index)[0][0]] = 1
            temp_mask[torch.where(sub_set_index)[0][1]] = 1
        red_node[i] = temp_mask
    return red_node


def hag_redun_search(mat):
    redun1 = torch.mm(mat, mat.T).fill_diagonal_(0)
    redun1[redun1 < 2] = 0
    indice1 = torch.where(redun1 != 0)
    red_mask1 = mat[indice1[0]] * mat[indice1[1]]
    unique = torch.unique(red_mask1, dim=0, return_inverse=True)[0]
    return unique


name = 'cora'
print("dataset：%s" % name)
matrix = torch.load("./data/original_adj_matrix/%s_matrix.pt" % name).int()

redun_mask = hag_redun_search(matrix)


def loop_of_match(adjmat, redundancy_mask, redundancy_node):
    hag_index = get_sorted(redundancy_mask)
    boost = 0
    remain = torch.clone(adjmat)
    redun_mat_list = []
    redun_node_list = []
    base = torch.zeros(size=[adjmat.shape[0]])
    for i in range(hag_index.shape[0]):
        ide = hag_index[i]
        node = torch.where(redundancy_node[ide] != 0)[0]  # 该冗余串能使用到的节点
        redun_mask = redundancy_mask[hag_index[i]]  # 该冗余串的掩码格式
        index = torch.where(redun_mask != 0)[0]
        is_subset = remain[node][:, index]
        available_node_num = torch.sum(torch.all(is_subset, dim=1))

        if available_node_num > 1:
            available_node = node[torch.all(is_subset, dim=1)]
            boost += (torch.sum(redun_mask)-1) * (available_node_num-1)
            remain[available_node] -= redun_mask
            redun_mat_list.append(redun_mask)
            node_mask = torch.clone(base)
            node_mask[available_node] = 1
            redun_node_list.append(node_mask)
    redun_mat = torch.tensor([aa.tolist() for aa in redun_mat_list])
    redun_node = torch.tensor([aa.tolist() for aa in redun_node_list])

    print("Boost of HAG: ", boost)
    return redun_mat, redun_node, remain


neighbor = get_neighbor_for_redun(redun_mask, matrix)
redun_mat, redun_node, remain=loop_of_match(matrix, redun_mask, neighbor)
torch.save(redun_mat,'./Rmat_Umat_QIT/HAG_%s_R_mat.pt'%name)
torch.save(redun_node,'./Rmat_Umat_QIT/HAG_%s_red_node.pt'%name)
torch.save(remain,'./Rmat_Umat_QIT/HAG_%s_U_mat.pt'%name)

"""
hag system intergrate
"""
hag_r_mat = redun_mask
