# QIT: Redundancy-eliminate-algorithm for GNN aggregation
GNN is widely used but faces a challenge in accelerating node aggregation due to inherent computational redundancy. 
This redundancy adds significant duplicates in forward computation and data transfers. 
We propose an efficient algorithm called QIT that introduces a redundancy string model to generalize groups of vertices used multiple times. 
By leveraging multiple levels of overlapping neighbors and many-to-many dependencies among overlapping nodes, our algorithm achieves finer-grained matching of overlapping neighbors and improves the searching process in the graph. 
This enables us to efficiently eliminate redundancy in the aggregation process for GNN, resulting in significant improvements in performance.
Our experiments with real-world graph datasets show that compared with SOTA--HAG, QIT improves redundancy elimination by 762%, improves speedup of GNN training tasks up to 359%, inference tasks up to 771%, and reduces memory overhead by over 77%.

This repo is consisting of three parts: (1) Redundancy Searching (2) Redundancy Matching (3) End to End.

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
QIT has four optional modes in end-to-end:
1. QIT inference
2. HAG infrence
3. QIT train
4. HAG train
