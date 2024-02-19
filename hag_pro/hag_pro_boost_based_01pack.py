import torch
import sys  # 导入sys模块

sys.setrecursionlimit(3000)
weight = torch.load('hag_pro_str_boost.pt')
exclusive = torch.load('hag_pro_redun_adj.pt')


def dfs(start, visited):
    global max_sum, path, exclusive_table, exclusive, max_path
    visited[start] = True
    path.append(start)
    is_leaf_flag = True
    index_exclu = torch.where(exclusive[start] == 0)[0]
    for i in range(index_exclu.shape[0]):
        exclusive_table.add(index_exclu[i].item())
    #print("node:",start,"exclu:",exclusive_table)
    if False in visited:
        not_visit_index = set([index for index, value in enumerate(visited) if not value])
        not_visit_index.difference_update(exclusive_table)

        available_index=list(not_visit_index)
        for nodes in range(len(available_index)):
            node = available_index[nodes]
            if node in exclusive_table:
                break
            else:
                is_leaf_flag = False
                dfs(node, visited)
    if is_leaf_flag:
        sumup = torch.sum(weight[path])
        if max_sum < sumup:
            max_sum = sumup
            max_path = path.copy()

    path.pop()


node_num = weight.shape[0]
global_max=0
global_path=[]
for start_node in range(node_num):
    max_sum = 0
    path, max_path = [], []
    exclusive_table = set()
    visited_nodes = [0] * node_num
    dfs(start_node, visited_nodes)
    if max_sum>global_max:
        global_max=max_sum
        global_path=max_path.copy()
        print("node:",start_node,"max sum:", global_max,"max path:", global_path)
