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

To evaluate other mothods, modify the code in utils/loader.py in lines 22-25 to execute other Redundancy-Elimination mode, as was done in End to End training and inference performance.



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



python ./hag_pro/hag_boost.py
```

