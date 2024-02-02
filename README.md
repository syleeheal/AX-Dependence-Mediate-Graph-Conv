# AX-Dependence-Mediates-Graph-Conv

This is the code repository of the work "Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective."
The datasets and codes to reproduce Observations, CSBM-X experiments, and feature shuffle experiments are provided.

---
## The Basics

The codes for algorithms and experiments are provided in **_./src/_** folder, including (_i_) the experiment pipeline, (_ii_) CFH measure in  ./src/measure.py, (_iii_) CSBM-X algorithm in ./src/csbm_x.py, and (_iv_) the feature shuffle algorithm in ./src/shuffle.py.

The **_./best_hyperparam/_** folder saves all the searched best hyperparameters. Running the experiments with ./src/main_shuffle.py will automatically load the best hyperparameters from the folder.

The **_./run/_** folder has the shell scripts to reproduce the experimental outcomes.

The **_./data/_** folder saves (_i_) the raw dataset and (_ii_) their Node2Vec vector.

The **_./results/_** folder saves all the experimental outcomes in CSV format.

We further provide the requirement.txt and random seeds used for the experiments.

---

## Base directory
```bash
cd ./AX-Dependence-Mediates-Graph-Conv/src
```

---

## Reproduce Observations

To reproduce Observations 1-2 in **_Figure 2_**, run the following shell script code.
```bash
sh ./run/observations.sh
```


---

## Reproduce CSBM-X Experiments
To reproduce the CSBM-X experiments in **_Figure 5_**, run the following shell script code.
```bash
sh ./run/csbmx_exps.sh
```

---

## Reproduce Feature Shuffle Experiments
To reproduce the feature shuffle experiments in **_Figure 6_**, run the following shell script code.
```bash
sh ./run/shuffle_exp(high_hc).sh
```

To reproduce the feature shuffle experiments in **_Figure 7_**, run the following shell script code.
```bash
sh ./run/shuffle_exp(low_hc).sh
```

To reproduce the feature shuffle experiments in **_Figure 8_**, run the following shell script code.
```bash
sh ./run/shuffle_exp(low_fd).sh
```

To reproduce the feature shuffle experiments in **_Figure 9_**, run the following shell script code.
```bash
sh ./run/shuffle_exp(other_gnns).sh
```

To reproduce the feature shuffle experiments in **_Figure 10_**, run the following shell script code.
```bash
sh ./run/shuffle_exp(proximity_feat).sh
```

To reproduce the feature shuffle experiments in **_Figure 16(b)_**, run the following shell script code.
```bash
sh ./run/shuffle_exp(sparse_split).sh
```

---

## Datasets
Executing the codes will automatically download the designated datasets from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). \
The codes to load the filtered **Chameleon** and **Squirrel** datasets, proposed by [Platonov et al. (2023, ICLR)](https://arxiv.org/pdf/2302.11640.pdf), are in _**filtered_dataset.py**_. \
The loading and preprocessing codes for each dataset are in _**src/utils.py**_. 


