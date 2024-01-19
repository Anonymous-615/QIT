import torch

def merge_muitl_redun():
    multi_redun=[]
    name=['cora','citeseer','ppi','pubmed']
    for id in range(4):
        for i in range(3):
            redun_mask=torch.load('%s_redun_level_%d.pt' % (name[id],i))
            multi_redun.append(redun_mask)
        redun_mask_all=torch.cat(multi_redun)
        torch.save(redun_mask_all,'%s_multi_redun.pt'%name[id])
    print("multi redun merge complete")