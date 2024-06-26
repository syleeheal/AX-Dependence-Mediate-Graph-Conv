# AX-Dependence-Mediates-Graph-Conv

This is the code repository of the work "[Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective](https://arxiv.org/abs/2402.04621)," published in ICML 2024. The datasets and codes to reproduce Observations, CSBM-X experiments, and feature shuffle experiments are provided.

---
## Visual Intuition of the Theoretical Results
![image](https://github.com/syleeheal/AX-Dependence-Mediate-Graph-Conv/assets/66083092/ad92555c-3d20-4a76-bc53-6725f8848d33)
_*** CFH stands for class-controlled feature homophily, and X|Y refers to class-controlled features._

---
## Basics

The codes for algorithms and experiments are provided in **_./src/_** folder, including (_i_) the experiment pipeline, (_ii_) CFH measure in  ./src/measure.py, (_iii_) CSBM-X algorithm in ./src/csbm_x.py, and (_iv_) the feature shuffle algorithm in ./src/shuffle.py.

The **_./best_hyperparam/_** folder saves all the searched best hyperparameters. Running the experiments with ./src/main_shuffle.py will automatically load the best hyperparameters from the folder.

The **_./run/_** folder has the shell scripts to reproduce the experimental outcomes.

The **_./data/_** folder saves (_i_) the raw dataset and (_ii_) their Node2Vec vectors.

The **_./results/_** folder saves all the experimental outcomes in CSV format.

We further provide the requirement.txt and random seeds used for the experiments.

---

## Base Directory
```bash
cd ./AX-Dependence-Mediates-Graph-Conv/src
```

---

## Reproducing Observations

To reproduce Observations 1-2 in **_Figure 2_**, run the following shell script code.
```bash
sh ./run/observations.sh
```


---

## Reproducing CSBM-X Experiments
To reproduce the CSBM-X experiments in **_Figure 5_** and **_Figure 16_**, run the following shell script code.
```bash
sh ./run/csbmx_exps.sh
```

---

## Reproducing Feature Shuffle Experiments
To reproduce the feature shuffle experiments, run the following shell script code.
```bash
sh ./run/shuffle_exp(high_hc).sh         ### reproduces Figure 6
sh ./run/shuffle_exp(low_hc).sh          ### reproduces Figure 7
sh ./run/shuffle_exp(low_fd).sh          ### reproduces Figure 8
sh ./run/shuffle_exp(other_gnns).sh      ### reproduces Figure 9
sh ./run/shuffle_exp(proximity_feat).sh  ### reproduces Figure 10
sh ./run/shuffle_exp(sparse_split).sh    ### reproduces Figure 17(b)
```

---

## Reproducing Pseudo-label-based Feature Shuffle Experiments
To reproduce the pseudo-label-based feature shuffle experiments in Table 3 of Appendix F, run the following shell script code.
```bash
sh ./run/pseudo_label_shuffle_exp.sh
```

---

## Datasets
Executing the codes will automatically download the designated datasets from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). \
The loading and preprocessing codes for each dataset are in _**src/utils.py**_. 


---

## Bibtex
```latex
@inproceedings{lee2024feature,
  title={Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective},
  author={Lee, Soo Yong and Kim, Sunwoo and Bu, Fanchen and Yoo, Jaemin and Tang, Jiliang and Shin, Kijung},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```

---

## Contacts
For any questions, please email me at syleetolow@kaist.ac.kr! 
