import os
import random
import argparse
import pickle

import torch
import pandas as pd
import numpy as np
from torch_geometric.utils import homophily

from model_opt import Trainer
from utils import load_graph, load_hyperparam, process_data, init_feat, print_outcome, save_results
from measure import CFH_measure
from shuffle import feature_shuffle, feature_noise
import gc




def main(args):

    # set seeds
    seeds = torch.tensor(torch.load('../seeds.pt'))[0].item()
    torch.manual_seed(seeds)
    random.seed(seeds)
    np.random.seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init graph
    graph_orig = load_graph(args)
    graph_orig = init_feat(args, graph_orig, init=args.feat_init) # init \in ['orig', 'n2v', 'adj', 'noise', 'ax']
    graph = graph_orig.clone()


    """
    graph stat eval
    """
    graph = feature_shuffle(args, args.shuffled_node_ratio, graph, graph.y, None, None, None)
    hc = homophily(graph.edge_index.cpu(), graph.y.cpu() , method='edge_insensitive')
    hg_norm, hvi_norm = CFH_measure(args, graph, count_self=False, measure='CFH')
    hg_norm_Y, hvi_norm_Y = CFH_measure(args, graph_orig, count_self=False, measure='class_homophily')
    hg_norm_X, hvi_norm_X = CFH_measure(args, graph_orig, count_self=False, measure='feat_homophily')

    print('Graph: {}'.format(args.dataset))
    print('CFH: {:.4f}'.format(hg_norm))
    print('Class Homophily: {:.4f}'.format(hc))
    print('Generalized Homophily (Class): {:.4f}'.format(hg_norm_Y))
    print('Generalized Homophily (Feature): {:.4f}'.format(hg_norm_X), '\n')

    statistics = [hc, hg_norm, hvi_norm, hg_norm_Y, hvi_norm_Y, hg_norm_X, hvi_norm_X, args.shuffled_node_ratio]
    
    if args.save == True: save_results('stat', args, statistics)

    del graph, graph_orig
    gc.collect()

    return statistics


def parameter_parser():
    
    parser = argparse.ArgumentParser()


    """DATASET"""
    parser.add_argument("--dataset", nargs="?", default="csbmx",)
    parser.add_argument("--feat-init", type=str, default="orig", )


    """CSBMX"""
    parser.add_argument("--num-nodes", type=int, default=10000, )
    parser.add_argument("--k", type=int, default=1, )
    parser.add_argument("--d_pos", type=int, default=10, )
    parser.add_argument("--d_neg", type=int, default=10, )
    parser.add_argument("--tau", type=float, default=0.0, )
    parser.add_argument("--mu", type=float, default=0.0, )
    parser.add_argument("--sigma", type=float, default=1.0, )


    """FEATURE"""
    parser.add_argument("--shuffled-node-ratio", type = float, default = 0)
    parser.add_argument("--noise-rate", type=float, default = 0)


    """EXPERIMENT"""
    parser.add_argument("--device", default="cuda:0", )
    parser.add_argument("--model", default="simple-gnn", )
    parser.add_argument("--save", type=bool, default=False,)

    return parser.parse_args()


if __name__ == "__main__":


    datasets = ['cora', 'citeseer', 'pubmed',  'cora_ml', 'cora-full','dblp',
                'wiki-cs',  'cs', 'physics',  'photo', 'computers', 'ogbn-arxiv', 
                'chameleon-filtered', 'squirrel-filtered', 'actor', 'texas', 'cornell', 'wisconsin', 
                'roman-empire', 'amazon-ratings', 'tolokers', 'penn94', 'flickr', 'ogbn-year']

    """run for all datasets"""
    if True:
        for d in datasets:
            args = parameter_parser()
            args.dataset = d
            statistics = main(args)