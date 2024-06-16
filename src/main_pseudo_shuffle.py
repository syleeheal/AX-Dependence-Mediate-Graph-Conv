import random
import argparse

import numpy as np
from torch_geometric.utils import homophily

from model_opt import Trainer
from utils import load_graph, load_hyperparam, process_data, init_feat, print_outcome, save_results
from measure import CFH_measure
from models import *




def main(args):

    """
    fit and eval model
    """
    # set seeds
    seeds = torch.tensor(torch.load('../seeds.pt'))[0].item()
    # seeds = random.randint(0, 100000)
    torch.manual_seed(seeds)
    random.seed(seeds)
    np.random.seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init graph
    graph_orig = load_graph(args)
    graph_orig = init_feat(args, graph_orig, init=args.feat_init)

    # init outputs
    train_accs, val_accs, test_accs, test_corrs = [], [], [], []

    hc = homophily(graph_orig.edge_index.cpu(), graph_orig.y.cpu() , method='edge_insensitive')
    hg_norm, hvi_norm = CFH_measure(args, graph_orig, count_self=False, measure='CFH')

    for exp in range(args.num_exp):

        #preprocess (data), init (split, trainer, model)
        graph, train_nodes, val_nodes, test_nodes = process_data(args, graph_orig)
        trainer = Trainer(args, graph, train_nodes, val_nodes, test_nodes)
        model = trainer.model_init()
        best_model = trainer.fit(graph, model)

        if args.pseudo_shuffle:
            graph_s = trainer.pseudo_shuffle(graph, best_model)
            best_model = trainer.fit(graph_s, best_model)
            train_acc, val_acc, test_acc, test_corr = trainer.eval(graph_s, best_model)
        else:
            train_acc, val_acc, test_acc, test_corr = trainer.eval(graph, best_model)
        
        # save results
        print_outcome(args, exp, test_acc, hg_norm, hc, swap_rate=None, print_summary=False)
        train_accs.append(train_acc), val_accs.append(val_acc), test_accs.append(test_acc), test_corrs.append(test_corr)
        if exp == (args.num_exp - 1): print_outcome(args, exp, test_accs, hg_norm, hc, swap_rate=None, print_summary=True)

    return test_accs

def parameter_parser():
    
    parser = argparse.ArgumentParser()


    """DATASET"""
    parser.add_argument("--dataset", nargs="?", default="csbmx",)
    parser.add_argument("--feat-init", type=str, default="orig", )


    """FEATURE"""
    parser.add_argument("--shuffled-node-ratio", type = float, default = 0)
    parser.add_argument("--noise-rate", type=float, default = 0)


    """EXPERIMENT"""
    parser.add_argument("--device", default="cuda:0", )
    parser.add_argument("--num-exp", default=30, type=int, help="Experiment number.")
    parser.add_argument("--model", default="gcn2", help="Model type.")
    parser.add_argument("--epochs", type=int, default=500, )
    parser.add_argument("--patience", type=int, default=100, )
    parser.add_argument("--split-type", type = str, default = 'count')
    parser.add_argument("--train-ratio", type = float, default = 20)
    parser.add_argument("--val-ratio", type = float, default = 30)
    parser.add_argument("--save", type=bool, default=False,)
    parser.add_argument("--data-hc", default="high")
    parser.add_argument("--pseudo-shuffle", type=bool, default=False, )


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
                'roman-empire', 'amazon-ratings', 'tolokers', 'penn94', 'flickr', 'arxiv-year']
    
    high_hc_datasets = ['citeseer','cora',  'pubmed',  'cora_ml', 'cora-full','dblp',
                        'wiki-cs',  'cs', 'physics',  'photo', 'computers', 'ogbn-arxiv', ]
    
    low_hc_datasets = ['chameleon-filtered', 'squirrel-filtered', 'actor', 'texas', 'cornell', 'wisconsin', 
                        'roman-empire', 'amazon-ratings', 'tolokers', 'penn94', 'flickr', 'arxiv-year']
    
    mixed_datasets = ['pubmed', 'cora-full', 'wiki-cs', 'computers', 'cs',  'ogbn-arxiv', 
                    'squirrel-filtered', 'actor', 'roman-empire', 'amazon-ratings', 'penn94', 'flickr']
    
    

    """run shuffle for all dataset"""
    if True:
        args = parameter_parser()

        if args.data_hc == 'high': datasets = high_hc_datasets
        elif args.data_hc == 'low': datasets = low_hc_datasets
        elif args.data_hc == 'mixed': datasets = mixed_datasets
        elif args.data_hc == 'all': datasets = all_datasets
        else: datasets = [args.dataset]

        for d in datasets:
            print(d)
            args.dataset = d
            args = load_hyperparam(args)
            test_accs = main(args)
            torch.cuda.empty_cache()