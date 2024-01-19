import time

import torch
"""
3冗余矩阵乘的算法执行时间优化
"""


def qit_search(mat):
    redun1 = torch.mm(mat, mat.T).fill_diagonal_(0)
    redun1[redun1 < 2] = 0
    indice1 = torch.where(redun1 != 0)
    red_mask1 = mat[indice1[0]] * mat[indice1[1]]
    unique=torch.unique(red_mask1, dim=0, return_inverse=True)[0]

    return unique


def hag_search(mat):

    redundancy=torch.zeros_like(mat)
    for i in range(mat.shape[0]):
        v1=i
        neighbor1=mat[v1]
        for j in range(mat.shape[1]):
            v2=j
            neighbor2 = mat[v2]
            redundancy[v1][v2]=torch.sum(neighbor1*neighbor2)
    #print("hag-search-time:",time.time()-start)


