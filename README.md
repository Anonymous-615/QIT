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

## Redundancy Searching


## Redundancy Matching


## End to End
