# QIT: Redundancy-Elimination system for GNN

QIT is an optimization tool library for GNN. It can find and eliminate redundancy based on the topological structure of the input data graph, thereby reducing the calculation and data handling of GNN. This process is accuracy-free.

QIT is compatible with other commonly used GNN optimization techniques such as quantization, sampling, sparsification, etc., and will not affect the correctness of the above methods, although the optimization effect may be diluted in some cases (such as sampling, which indirectly changes the topology of the graph structure, thereby affecting the optimization quality of QIT).

This repo is consisting of four parts: (1) End to End training and inference performance. (2) Redundancy-Elimination performance. (3) Heuristic performance. (4) Match performance.


##Requirements
```
    pip install pytorch=1.11.0=py3.8_cuda11.3_cudnn8_0
    pip install pyg=2.0.4=py38_torch_1.11.0_cu113
```

If the environment is configured incorrectly, try:

```
conda env create -f environment.yml
```

## End to End training and inference performance

To run the end-to-end performance of QIT:
```
python main
```

Modify the code in utils/loader.py in lines 22-25 to execute other Redundancy-Elimination mode. For example, to run HAG based GNN end-to-end task:
```
redun_free_edge_index = redun_eliminate_hag(data).to(device) 
```

Then run the code:
```
python main
```



## Redundancy-Elimination performance
We evaluate the redundancy elimination rate of the GNN aggregation operator by comparing the change in the total edge number of the graph before and after redundancy elimination.
The formula is：100% \times (len(redun_eli_edge_index)/len(origin_edge_index))

To get the boosts of qit, uncomment lines 14-18 of main.py:

```
    # optional:Redundancy-Elimination Performance
    origin_edge=datas[0].edge_index
    redun_eli_edge=datas[3]
    boost=(1-len(redun_eli_edge)/len(origin_edge))*100
    print('Redundancy-Elimination Boost: %.2f'%boost)
```







## Heuristic performance
To match the Redundancy Set and adjacency matrix, QIT has several optional modes:
1. Many-to-many
2. 1-to-1
3. Length based sort
4. Contribution based sort ( length * times)
5. other sort mode

To run many-to-many mode, use:
```
python ./match/qit_match.py
```
To run 1-to-1 mode, use:
```
python ./match/hag_match.py
```


## Match performance
QIT has several optional modes in end-to-end:
1. QIT inference:
```
python ./end-to-end/end_to_end.py
```
2. HAG infrence:
```
python ./end-to-end/end_to_end_hag.py
```
3. QIT/HAG train:
```
python ./end-to-end/end_to_end_train.py
```

If you want to try different device, change torch.device to 'cuda'. If you want to test different aggregation modes, change the 'flag' variable in the code.Flags from 1 to 4 correspond to: 
- GEMM-based primitive aggregation 
- GEMM-based QIT/HAG aggregation 
- SPMM-based primitive aggregation 
- SPMM-based QIT/HAG aggregation respectively


## Compared to HAG and HAG-PRO

The improved version of HAG (hereinafter referred to as HAG-pro) announced that they have enhanced the redundant matching algorithm of HAG using a partial greedy approach, thereby improving the overall redundancy elimination effectiveness. The paper can be found at （https://ieeexplore.ieee.org/abstract/document/9517814）.

According to our experiment results, HAG-PRO cannot guarantee that its redundancy elimination effectiveness is strictly superior to HAG. HAG-PRO equates the redundancy matching problem to maximum weighted-hypergraph matching problems and compromises algorithm execution speed by seeking an approximate solution to this problem as the matching result. We exhaustively searched for all solutions on smaller datasets and obtained the globally optimal solution. Experimental results indicate that in the cora and citeseer datasets, the redundancy elimination effectiveness of HAG-PRO is inferior to that of HAG. This is mainly because in HAG-PRO, if two redundant strings are mutually exclusive, then they cannot be adopted simultaneously. However, in HAG and QIT, redundant strings can only be unable to simultaneously accelerate a subtask when they are mutually exclusive for that specific subtask.

Therefore, in our paper, QIT is not compared with HAG-PRO, but the more effective HAG is used as our benchmark.

To run our replicated redundancy elimination experiments of HAG-PRO:

1. Find redundancy, compute hyperedge weights, get mutual exclusion between hyperedges:
```
python ./hag_pro/preprocess.py
```
2. DFS-based maximum weighted-hypergraph matching:
```
python ./hag_pro/hag_pro_boost_based_01pack.py
```
3. As a comparison, to run HAG's redundant matching results:
```
python ./hag_pro/hag_boost.py
```

