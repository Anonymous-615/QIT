import torch


def get_neighbor_for_redun(red_mask, mat):

    base = torch.zeros(size=[mat.shape[0]], dtype=bool)
    red_node = torch.zeros_like(red_mask, dtype=bool)

    for i in range(red_mask.shape[0]):
        redun=red_mask[i]
        redun_id = torch.where(redun)[0]  # red.shape=[2708]
        neighbors=mat[redun_id]
        id_of_neighbors=torch.unique(torch.where(neighbors)[1])
        intersections=torch.mul(mat[id_of_neighbors],redun)
        available_neighbors_index=torch.where(torch.sum(intersections,dim=1)>1)[0]
        available_neighbors_id=id_of_neighbors[available_neighbors_index]
        temp_mask = torch.clone(base)
        temp_mask[available_neighbors_id] = 1
        red_node[i] = temp_mask
    #torch.save(red_node, 'pubmed_red_node.pt')
    return red_node


# redun_mask=torch.load('../multi_level/pubmed_multi_redun_all.pt')
# matrix=torch.load('../matrix/pubmed_matrix.pt')
# print("冗余集规模为：",redun_mask.shape[0])
# get_neighbor_for_redun(redun_mask,matrix)