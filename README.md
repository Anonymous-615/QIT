# QIT: Redundancy-eliminate-algorithm for GNN aggregation
GNN is widely used but faces a challenge in accelerating node aggregation due to inherent computational redundancy. 
This redundancy adds significant duplicates in forward computation and data transfers. 
We propose an efficient algorithm called QIT that introduces a redundancy string model to generalize groups of vertices used multiple times. 
By leveraging multiple levels of overlapping neighbors and many-to-many dependencies among overlapping nodes, our algorithm achieves finer-grained matching of overlapping neighbors and improves the searching process in the graph. 
This enables us to efficiently eliminate redundancy in the aggregation process for GNN, resulting in significant improvements in performance.
Our experiments with real-world graph datasets show that compared with SOTA--HAG, QIT improves redundancy elimination by 762%, improves speedup of GNN training tasks up to 359%, inference tasks up to 771%, and reduces memory overhead by over 77%.

This repo is consisting of four parts: (1) Redundancy Searching (2) Redundancy Matching (3) End to End (4) Compared to HAG-PRO.

**Requirements**


```
    pip install pytorch=1.11.0=py3.8_cuda11.3_cudnn8_0
    pip install pyg=2.0.4=py38_torch_1.11.0_cu113
```

## Get start
To get started, you need to download datasets and extract the corresponding adjacency matrix from them:
```
python ./dataset/get_dataset.py
```


## Redundancy Searching
To search for the Redundancy Set from datasets above, QIT has several optional modes:

1. GEMM based search
2. Iterator based search
3. Multi-level search
4. Single level search

Modify the code of search/multi_redun.py in lines 14 and 15 to execute the corresponding pattern. For example:
```
search_mode='gemm'
search_level='single'
```
Then run the code:
```
python ./search/multi_redun.py
```
So you can get the single-layer redundancy set by gemm based search.
## Redundancy Matching
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


## End to End
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


## Compared to HAG-PRO

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

