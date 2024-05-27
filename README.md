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
python main.py
```

Modify the code in utils/loader.py in lines 22-25 to execute other Redundancy-Elimination mode. For example, to run HAG based GNN end-to-end task:
```
redun_free_edge_index = redun_eliminate_hag(data).to(device) 
```

Then run the code:
```
python main.py
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
We evaluate the redundancy elimination rate on different heuristics by comparing the change in the total edge number of the graph before and after redundancy elimination.
Supported heuristics are:
(1)QIT
(2)HAG
(3)HAG-PRO
(4)Random
(5)Default order

To change the heus, modify lines 53-57 of redun_elimination/match.py. For example, to run the redundancy-elimination by QIT-heu:

```
name = 'conflict'
```

Or to run the redundancy-elimination by HAG-heu:
```
name = 'hag'
```

## Match performance
We evaluate the redundancy elimination rate on different redundancy matching mode by comparing the change in the total edge number of the graph before and after redundancy elimination.
Supported Match modes are:
(1)QIT-Match
(2)HAG-Match
(3)HAG-PRO-Mode1-Match
(4)HAG-PRO-Mode2-Match

To change the Match mode, first uniformly change heus to RAND for different methods, including loop_of_match(), hag_search_and_match(), hag_pro_search_and_match(), hag_pro_HUB_search_and_match() in redun_elimination/match.py. Take hag_search_and_match() for example (line 153):

```
index = torch.randperm(unique.size(0))
```



Then modify the code in utils/loader.py in lines 22-25 to execute different Match mode of Redundancy-Elimination. For example, to run HAG based GNN end-to-end task:
```
redun_free_edge_index = redun_eliminate_hag(data).to(device) 
```

