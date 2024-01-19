import torch


def get_sorted(mask, neighbor,name):

    redun_len = torch.sum(mask, dim=1)
    redun_node_len = torch.sum(neighbor, dim=1)
    if name == 'flip':
        return torch.flip(torch.arange(torch.sum(mask, dim=1).shape[0]), dims=[0])
    if name == 'order':
        return torch.arange(torch.sum(mask, dim=1).shape[0])

    if name == 'rand':
        original_sequence = torch.arange(torch.sum(mask, dim=1).shape[0])
        shuffled_indices = torch.randperm(original_sequence.size(0))
        shuffled_sequence = original_sequence[shuffled_indices]
        return shuffled_sequence

    if name == 'hag':
        return torch.sort(redun_len, descending=False)[1]
    if name == 'qit':
        return torch.sort(redun_len * redun_node_len, descending=True)[1]
