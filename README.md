# DHyper

This is the source code for paper [A Recurrent Dual Hypergraph Neural Network for Event Prediction in Temporal Knowledge Graphs]


## Data
We used three public datasets collected in [3]:

GDELT18

ICEWS18

ICEWS14

We processed these three datasets and got three country based datasets:

GDELT18C

ICEWS18C

ICEWS14C


## Prerequisites

- Python 3.7.7

- PyTorch 1.6.0

- dgl 0.5.2

- Sklearn 0.23.2

- Pandas 1.1.1

- tqdm

- pandas

- rdflib



## Training and testing

Please run following commands for training and testing. We take the dataset `ICEWS14s` as the example.

**Event prediction**

python:

python main.py -d ICEWS14s --train-history-len 7 --test-history-len 1 --dilate-len 1 --lr 0.01 --n-layers 3 --evaluate-every 1  --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --group_num 14 --lgroup_num 8 --gpu 0




## Citation

Please cite our paper if you find this code useful for your research.



## References

[1]	Kalev Leetaru and Philip A. Schrodt. 2013. GDELT: Global data on events, location, and tone, 1979-2012. ISA Annual Convention, 2(4): 1-49.

[2]	Elizabeth Boschee, Jennifer Lautenschlager, Sean O’Brien, Steve Shell-man, James Starz, and Michael Ward. 2015. ICEWS coded event data. Harvard Dataverse.

[3]	Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. 2021. Temporal knowledge graph reasoning based on evolutional representation learning. In SIGIR, 408-417.