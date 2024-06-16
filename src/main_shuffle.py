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
import pdb




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


    # init outputs
    train_accs, val_accs, test_accs, test_corrs = [], [], [], []
    hc_list, hg_list, hvi_list = [], [], []
    shuffled_node_ratio_list = []


    hc = homophily(graph.edge_index.cpu(), graph.y.cpu() , method='edge_insensitive')
    hg_norm, hvi_norm = CFH_measure(args, graph, count_self=False, measure='CFH')

    """
    shuffle eval
    """
    for exp in range(args.num_exp):

        num_shuffles = 100
        graph, train_nodes, val_nodes, test_nodes = process_data(args, graph_orig)
        graph = feature_noise(args, args.noise_rate, graph)

        trainer = Trainer(args, graph, train_nodes, val_nodes, test_nodes)

        for i in range(num_shuffles+1):

            shuffled_node_ratio = (1 / num_shuffles) * (i)
            graph_s = feature_shuffle(args, shuffled_node_ratio, graph, graph.y, train_nodes, val_nodes, test_nodes)
            hg_norm, hvi_norm = CFH_measure(args, graph_s, count_self=False, measure='CFH')

            model = trainer.model_init()
            best_model = trainer.fit(graph_s, model)
            train_acc, val_acc, test_acc, test_corr = trainer.eval(graph_s, best_model)

            # save results 
            train_accs.append(train_acc), val_accs.append(val_acc), test_accs.append(test_acc), test_corrs.append(test_corr)
            hc_list.append(hc), hg_list.append(hg_norm), hvi_list.append(hvi_norm)
            print_outcome(args, i, test_acc, hg_norm, hc, shuffled_node_ratio, print_summary=False)
            shuffled_node_ratio_list.append(shuffled_node_ratio)
            
            outputs = [train_accs, val_accs, test_accs, test_corrs, hc_list, hg_list, hvi_list, shuffled_node_ratio_list]
 

    train_accs, val_accs, test_accs, test_corrs, hc_list, hg_list, hvi_list, shuffled_node_ratio_list = outputs    
    print_outcome(args, exp, test_accs, hg_list, hc_list, shuffled_node_ratio_list, print_summary=True)

    if args.save == True: save_results('shuffle', args, outputs)

    return test_accs


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
    parser.add_argument("--num-exp", default=5, type=int, help="Experiment number.")
    parser.add_argument("--model", default="gcn2", help="Model type.")
    parser.add_argument("--epochs", type=int, default=500, )
    parser.add_argument("--patience", type=int, default=100, )
    parser.add_argument("--split-type", type = str, default = 'ratio')
    parser.add_argument("--train-ratio", type = float, default = 0.5)
    parser.add_argument("--val-ratio", type = float, default = 0.25)
    parser.add_argument("--save", type=bool, default=False,)
    parser.add_argument("--data-hc", default="high", help="Tested dataset class homophily.")


    """HYPER-PARAMETERS"""
    parser.add_argument("--iterations", type=int, default=10,) 
    parser.add_argument("--num-layers", type = int, default = 1, ) 
    parser.add_argument("--dropout", type=float, default=0.5,)

    parser.add_argument("--lr", type=float, default=0.01, )
    parser.add_argument("--dr", type=float, default=0.0005,) 

    parser.add_argument("--hid-dim", type=int, default=64,) 
    parser.add_argument("--alpha", type=float, default=0.5, ) 
    parser.add_argument("--lambd", type=float, default=1, ) 

    return parser.parse_args()


if __name__ == "__main__":


    all_datasets = ['cora', 'citeseer', 'pubmed',  'cora_ml', 'cora-full','dblp',
                'wiki-cs',  'cs', 'physics',  'photo', 'computers', 'ogbn-arxiv', 
                'chameleon-filtered', 'squirrel-filtered', 'actor', 'texas', 'cornell', 'wisconsin', 
                'roman-empire', 'amazon-ratings', 'tolokers', 'penn94', 'flickr', 'ogbn-year']
    
    high_hc_datasets = ['cora', 'citeseer', 'pubmed',  'cora_ml', 'cora-full','dblp',
                        'wiki-cs',  'cs', 'physics',  'photo', 'computers', 'ogbn-arxiv', ]
    
    low_hc_datasets = ['chameleon-filtered', 'squirrel-filtered', 'actor', 'texas', 'cornell', 'wisconsin', 
                        'roman-empire', 'amazon-ratings', 'tolokers', 'penn94', 'flickr', 'ogbn-year']
    
    mixed_datasets = ['pubmed', 'cora-full', 'wiki-cs', 'computers', 'cs',  'ogbn-arxiv', 
                    'squirrel-filtered', 'actor', 'roman-empire', 'amazon-ratings' 'penn94', 'flickr']

    """run shuffle for all dataset"""
    if True:
        args = parameter_parser()

        if args.data_hc == 'high': datasets = high_hc_datasets
        elif args.data_hc == 'low': datasets = low_hc_datasets
        elif args.data_hc == 'mixed': datasets = mixed_datasets
        else: datasets = all_datasets

        for d in datasets:
            args.dataset = d
            args = load_hyperparam(args)
            test_accs = main(args)
            torch.cuda.empty_cache()