import torch
from gemm_based_search import qit_search,hag_search
from merge import merge_muitl_redun


def iterate(mat):
    for i in range(3):
        a = qit_search(mat)
        torch.save(a,'./output/%s_redun_level_%d.pt'%(name[id],i))
        mat = a


name=['cora','citeseer','ppi','pubmed']
search_mode='gemm'#'iter'
search_level='multi'#'single'

for id in range(4):
    print("dataset:",name[id])
    matrix = torch.load('../data/origin-adj-matrix/%s_matrix.pt'%name[id]).float()
    if search_level=='single':
        if search_mode=='gemm':
            torch.save(qit_search(matrix), './output/%s_redun.pt' % name[id])
        else:
            torch.save(hag_search(matrix), './output/%s_redun.pt' % name[id])
    if search_level=='multi':
        iterate(matrix)
        merge_muitl_redun()
    print("dataset search complete")
    print()

