import os
import pandas as pd
from torch import Tensor
import random
from tqdm import trange, tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Amazon, WikiCS, WebKB, WikipediaNetwork, Actor, LINKXDataset, Flickr, Coauthor, CitationFull, CoraFull, HeterophilousGraphDataset, AttributedGraphDataset 
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, homophily
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_std, scatter_max, scatter_min, scatter_add
from torch_geometric.nn.models import Node2Vec

from csbm_x import csbmx_graph, csbmx_two_graph
from measure import CFH_measure
from ogb.nodeproppred import PygNodePropPredDataset
from tabulate import tabulate
from models import Propagation_Layer
import pdb

def load_graph(args):
    
    path = '../data/'
    
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        graph = Planetoid(root=path, name=args.dataset.capitalize(), split = 'public')[0]
    
    elif args.dataset == 'wiki-cs':
        graph = WikiCS(root=path + 'Wiki-CS', is_undirected=True)[0]

    elif args.dataset in ['photo', 'computers']:
        graph = Amazon(root=path, name=args.dataset.capitalize())[0]

    elif args.dataset in ['dblp', 'cora_ml']:
        graph = CitationFull(root=path, name=args.dataset.capitalize())[0]

    elif args.dataset == 'cora-full':
        graph = CoraFull(root=path + 'cora-full')[0]

    elif args.dataset in ['texas', 'cornell', 'wisconsin']:
        graph = WebKB(root=path, name=args.dataset.capitalize())[0]

    elif args.dataset in ['squirrel', 'chameleon']:
        graph = WikipediaNetwork(root=path, name=args.dataset.capitalize())[0]

    elif args.dataset == 'actor':
        graph = Actor(root=path + 'actor')[0]

    elif args.dataset in ['chameleon-filtered', 'squirrel-filtered']:
        graph = torch.load(path + args.dataset + '.pt')

    elif args.dataset in ['penn94', 'genius']:
        graph = LINKXDataset(root=path, name=args.dataset.capitalize())[0]

    elif args.dataset == 'flickr':
        graph = Flickr(root=path + 'flickr')[0]
    
    elif args.dataset in ["cs", "physics"]:
        graph = Coauthor(root=path, name=args.dataset)[0]

    elif args.dataset in ["roman-empire", "amazon-ratings", "tolokers"]:
        graph = HeterophilousGraphDataset(root=path, name=str(args.dataset).capitalize())[0]

    elif args.dataset == 'ogbn-arxiv':
        graph = PygNodePropPredDataset(name='ogbn-arxiv', root=path)[0]
        graph.y = graph.y.squeeze()

    elif args.dataset == 'arxiv-year':
        graph = PygNodePropPredDataset(name='ogbn-arxiv', root=path)[0]
        label = even_quantile_labels(graph.node_year.flatten(), nclasses=5, verbose=False)
        graph.y = torch.as_tensor(label).long()

    elif args.dataset in ['csbmx']:
        graph = csbmx_graph(args)

    elif args.dataset in ['csbmx2']:
        graph = csbmx_two_graph(args)

    return graph


def load_hyperparam(args):
    
    path = '../best_hyperparam/'  + args.feat_init + '/' + args.model + '_' + args.dataset + '_' + str(args.train_ratio) + '.txt'

    if os.path.isfile(path):

        with open(path, 'r') as f: 
            
            for i in range(4): 
                line = f.readline()
            
            line = line[23:-2]

            line = line.replace(',', ' ')
            num_hyperparam = int((len(line.split('\''))-1) / 2)

            for i in range(num_hyperparam):
                strings = line.split('\'')[1:][i*2]
                value = line.split('\'')[1:][(i*2)+1].split()[1]
                
                args.__dict__[strings] = float(value)

    else:
        print('No hyperparameter file found. Using default hyperparameters.')

    return args


def split_per_label_ratio(graph, train_ratio, val_ratio):

    num_nodes = graph.x.size(0)
    num_labels = int(graph.y.max() + 1)

    nodes = torch.arange(num_nodes)
    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(num_labels):
        nodes_i = nodes[graph.y == i]
        nodes_i = nodes_i[random.sample(range(nodes_i.shape[0]), nodes_i.shape[0])]
        num_train = int(nodes_i.shape[0] * train_ratio)
        num_val = int(nodes_i.shape[0] * val_ratio)

        half_nodes = int(nodes_i.shape[0] / 2)
        quarter_nodes = int(nodes_i.shape[0] / 4)

        train_indices.append(nodes_i[0 : half_nodes][:num_train])
        val_indices.append(nodes_i[half_nodes : half_nodes+quarter_nodes][:num_val])
        test_indices.append(nodes_i[half_nodes+quarter_nodes : ])

    train_idx = torch.cat(train_indices)
    val_idx = torch.cat(val_indices)
    test_idx = torch.cat(test_indices)

    
    return train_idx, val_idx, test_idx


def split_per_label_count(graph, train_num_per_label, val_num_per_label):

    num_nodes = graph.x.size(0)
    num_labels = int(graph.y.max() + 1)

    nodes = torch.arange(num_nodes)
    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(num_labels):
        nodes_i = nodes[graph.y == i]
        nodes_i = nodes_i[random.sample(range(nodes_i.shape[0]), nodes_i.shape[0])]

        train_indices.append(nodes_i[0 : train_num_per_label])
        val_indices.append(nodes_i[train_num_per_label : train_num_per_label + val_num_per_label])
        test_indices.append(nodes_i[train_num_per_label + val_num_per_label : ])

    train_idx = torch.cat(train_indices)
    val_idx = torch.cat(val_indices)
    test_idx = torch.cat(test_indices)

    
    return train_idx, val_idx, test_idx


def process_data(args, graph):

    graph = graph.clone()


    """
    data split
    """
    if args.split_type == 'count':
        split = split_per_label_count(graph, train_num_per_label=args.train_ratio, val_num_per_label=args.val_ratio)
    elif args.split_type == 'ratio':
        split = split_per_label_ratio(graph, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_nodes, val_nodes, test_nodes = split


    """
    preprocess
    """
    if args.model not in ['simple-gnn']:
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
        graph = transform(graph)


    """
    to GPU
    """
    graph = graph.to(args.device)


    return graph, train_nodes, val_nodes, test_nodes


def init_feat(args, graph, init):

    graph = graph.clone()

    if init == 'orig':
        if args.model =='aero':
            if args.dataset in ['computers', 'photo', 'flickr']:
                graph.x = F.normalize(graph.x, p=1, dim=1)
        pass

    elif init == 'n2v':
        
        path = '../data/node2vec/' + args.dataset + '.pt'
        if (args.dataset + '.pt') not in os.listdir('../data/node2vec/'):
            print('Node2Vec training...:', args.dataset)
            model = Node2Vec(graph.edge_index, embedding_dim=256, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, sparse=True).to(args.device)
            loader = model.loader(batch_size=256, shuffle=True, num_workers=4)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
            iterator = trange(1, 100, desc='Loss: ', leave=False)
            model.train()
            for _ in iterator:
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(args.device), neg_rw.to(args.device))
                    loss.backward()
                    optimizer.step()
                    iterator.set_description("Loss: {:.4f}".format(loss.item()))
            model.eval()
            n2v = model.embedding.weight.detach().cpu()
            torch.save(n2v, path)
        else:
            n2v = torch.load(path)

        graph.x = n2v

    elif init == 'adj':
        adj = graph.edge_index
        adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), (graph.num_nodes, graph.num_nodes)).to_dense().float()
        adj = adj + torch.eye(adj.shape[0])
        adj = adj / adj.sum(dim=1, keepdim=True)
        graph.x = adj 

    elif init == 'noise':
        gaussian_sampler = torch.distributions.normal.Normal(0, 1)
        noise = gaussian_sampler.sample((graph.x.shape[0], 1)).to(args.device)
        graph.x = noise

    elif init == 'ax':
        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
        graph_und = transform(graph).clone().to(args.device)
        prop_layer = Propagation_Layer(K=args.num_layers).to(args.device)
        with torch.no_grad(): graph.x = prop_layer(graph_und.x, graph_und.edge_index)

    return graph
    

def save_results(task, args, outputs):

    if task in ['csbmx', 'csbmx2']:
        train_accs, val_accs, test_accs, test_corrs, hc_list, hg_list, hvi_list, shuffled_node_ratio_list = outputs
        results_df = pd.DataFrame({'model': args.model,
                                    'dataset': args.dataset,
                                    'num-nodes': args.num_nodes,
                                    'mu': args.mu,
                                    'd_pos': args.d_pos,
                                    'd_neg': args.d_neg,
                                    't': args.tau,
                                    'k': args.k,
                                    'sigma': args.sigma,
                                    'train_acc': np.mean(train_accs),
                                    'val_acc': np.mean(val_accs),
                                    'test_acc': np.mean(test_accs),
                                    'CFH': np.mean(hg_list),
                                    'class-homophily': np.mean(hc_list),
                                    'train-ratio': args.train_ratio,
                                    'val-ratio': args.val_ratio,
                                    'num-exp': args.num_exp,
                                    'num-layers': args.num_layers,
                                    },
                                    index=[0])
        results_df.set_index('dataset', inplace=True)

        path = '../results/{}/{}_{}.csv'.format(args.dataset, args.model, args.mu)
        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
        if os.path.exists(path): results_df.to_csv(path, mode='a', header=False)
        else: results_df.to_csv(path, mode='w', header=True)
    
    elif task == 'shuffle':
        train_accs, val_accs, test_accs, test_corrs, hc_list, hg_list, hvi_list, shuffled_node_ratio_list = outputs
        length = len(train_accs)
        results_df = pd.DataFrame({'model': np.repeat(args.model, length), 
                                   'dataset': np.repeat(args.dataset, length), 
                                   'train_acc': train_accs,
                                   'val_acc': val_accs, 
                                   'test_acc': test_accs,
                                   'CFH': hg_list, 
                                   'class-homophily': hc_list, 
                                   'shuffled-node-ratio': shuffled_node_ratio_list, 
                                   'num-exp': np.repeat(args.num_exp, length),
                                   'train-ratio': np.repeat(args.train_ratio, length),
                                   'noised-node-ratio': np.repeat(args.noise_rate, length),
                                   'feature-type': np.repeat(args.feat_init, length),
                                   })
        results_df.set_index('dataset', inplace=True)
        results_df = results_df.groupby(['dataset', 'model', 'shuffled-node-ratio', 'feature-type']).mean()

        path = '../results/shuffle/{}/{}.csv'.format(args.dataset, args.model)
        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
        if os.path.exists(path): results_df.to_csv(path, mode='a', header=False)
        else: results_df.to_csv(path, mode='w', header=True)

    elif task == 'stat':

        hc, hg_norm, hvi_norm, hg_norm_Y, hvi_norm_Y, hg_norm_X, hvi_norm_X, shuffled_node_ratio = outputs

        hc =  [hc]
        hg_norm = [hg_norm]
        hvi_norm = [hvi_norm]
        hg_norm_Y = [hg_norm_Y]
        hvi_norm_Y = [hvi_norm_Y]
        hg_norm_X = [hg_norm_X]
        hvi_norm_X = [hvi_norm_X]
        shuffled_node_ratio = [shuffled_node_ratio]

        results_df = pd.DataFrame({'dataset': args.dataset,
                                    'feature-type': args.feat_init,
                                    'shuffled-node-ratio': shuffled_node_ratio, 
                                    'class-homophily': hc,
                                    'CFH(graph)': hg_norm,
                                    'CFH(node)': hvi_norm,
                                    'generalized-H(Y, graph)': hg_norm_Y,
                                    'generalized-H(Y, node)': hvi_norm_Y,
                                    'generalized-H(X, graph)': hg_norm_X,
                                    'generalized-H(X, node)': hvi_norm_X,
                                    },
                                    index=[0])

        path = '../results/stat/{}.csv'.format(args.dataset)
        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
        if os.path.exists(path): results_df.to_csv(path, mode='a', header=False)
        else: results_df.to_csv(path, mode='w', header=True)


def print_outcome(args, exp, test_acc, hg_norm, hc, swap_rate, print_summary):

    if print_summary:

        test_acc = np.mean(test_acc)
        hg_norm = np.mean(hg_norm)
        hc = np.mean(hc)
        exp = int(exp + 1)

        print('\n Results Summary:')
        print(tabulate([[exp, test_acc, hg_norm, hc]], 
                       headers=['Num-Trials', 'Test Acc', 'h(G)', 'hc'], 
                       tablefmt='orgtbl',
                       stralign='center',
                       numalign='center',
                       floatfmt='.4f',
                       ))
        print('\n')
        


    else:

        if args.dataset in ['csbmx', 'csbmx2']:
            if exp == 0: print(tabulate([['Trial', 'Test Acc', 'h(g)', 'hc', 'tau', 'd+/d-', 'mu', 'n']], tablefmt='orgtbl'))
            print(tabulate([[exp, test_acc, hg_norm, hc, args.tau, (args.d_pos / args.d_neg), args.mu, args.num_nodes]], 
                           tablefmt='orgtbl', 
                           stralign='center', 
                           numalign='center', 
                           floatfmt='.4f',
                           ))

        else:
            if exp == 0: print(tabulate([['Trial', 'Test Acc', 'h(g)', 'hc', 'Shuffled Node Ratio']], tablefmt='orgtbl'))
            print(tabulate([[exp, str(test_acc), hg_norm, hc, swap_rate]], 
                           tablefmt='orgtbl', 
                           stralign='center', 
                           numalign='center', 
                           floatfmt='.4f',
                           ))


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int32)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label
