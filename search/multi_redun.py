import torch
from gemm_based_search import qit_search
from merge import merge_muitl_redun


def iterate(mat):
    for i in range(3):
        a = qit_search(mat)
        torch.save(a,'./output/%s_level_%d.pt'%(name[id],i))
        mat = a


name=['cora','citeseer','ppi','pubmed']
for id in range(4):
    print("dataset:",name[id])
    matrix = torch.load('../data/origin-adj-matrix/%s_matrix.pt'%name[id]).float()
    iterate(matrix)
    print("dataset search complete")
    print()
merge_muitl_redun()
