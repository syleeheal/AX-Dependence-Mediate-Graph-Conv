import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, homophily
from torch_scatter import scatter_mean
from tqdm import tqdm
import gc



def CFH_measure(args, graph, count_self=False, measure='CFH'):

    Y = graph.y.clone()
    X = graph.x.clone()
    edge_index = graph.edge_index.clone()


    """preprocess"""
    c = Y.max().item() + 1 
    c_index = [Y == label for label in range(c)]

    edge_index = remove_self_loops(graph.edge_index)[0]

    if measure == 'CFH': 
        for label in range(c):
            mean = X[c_index[label]].mean(dim=0, keepdim=True)
            X[c_index[label]] = X[c_index[label]] - mean
        X[torch.isnan(X)] = 0

    elif measure == 'class_homophily': 
        idx = Y == -1
        Y[idx] = 0
        if len(Y.shape) == 2: X = Y.float()
        if len(Y.shape) == 1: 
            X = F.one_hot(Y, num_classes=c).float()
            X[idx] = 0

    elif measure == 'feat_homophily': 
        pass

    if count_self == True: edge_index = add_self_loops(edge_index, num_nodes=X.shape[0])[0]


    """to gpu"""
    X = X.to(args.device)
    Y = Y.to(args.device)
    edge_index = edge_index.to(args.device)
    

    """calc d_vij"""
    d_vij = []
    batch_size = 1000
    for i in tqdm(range(0, edge_index.shape[1], batch_size), leave=False):
        src_x = X[edge_index[0, i:i+batch_size]]
        dst_x = X[edge_index[1, i:i+batch_size]]
        d_vij.append(torch.norm(src_x - dst_x, p=2, dim=1)) 
    d_vij = torch.cat(d_vij, dim=0)
    d_vi_Ni = scatter_mean(d_vij, edge_index[0], dim=0, dim_size=X.shape[0]) 


    """calc b(v_i)"""
    b_vi = []
    batch_size = 1000
    for i in tqdm(range(0, X.shape[0], batch_size), leave=False):
        _b_vi = torch.cdist(X[i:i+batch_size], X, p=2)
        if count_self == False: 
            _b_vi = _b_vi.sum(dim=1) / (_b_vi.shape[1] - 1)
        else: 
            _b_vi = _b_vi.mean(dim=1)
        b_vi.append(_b_vi)
    b_vi = torch.cat(b_vi, dim=0)


    """measure CFH_h_pij"""
    h_pij = b_vi[edge_index[0]] - d_vij 
    

    """measure CFH_h_vi"""
    h_vi = scatter_mean(h_pij, edge_index[0], dim=0, dim_size=X.shape[0]) 


    """normalize CFH_h_vi"""
    pos_idx, neg_idx = h_vi >= 0, h_vi < 0
    h_vi_norm = h_vi.clone()
    h_vi_norm[pos_idx] = h_vi_norm[pos_idx] / b_vi[pos_idx]
    h_vi_norm[neg_idx] = h_vi_norm[neg_idx] / d_vi_Ni[neg_idx]


    """measure CFH_h_g"""
    h_g = h_vi.mean()


    """normalize CFH_h_g"""
    h_g_norm = h_g.clone()
    if h_g_norm >= 0: h_g_norm = (h_g_norm / b_vi.mean().item()).item()
    if h_g_norm < 0: h_g_norm = (h_g_norm / d_vi_Ni.mean().item()).item()
    h_g = h_g.item()


    h_vi = h_vi.detach().cpu().tolist()
    h_vi = [round(elem, 4) for elem in h_vi]
    h_vi_norm = h_vi_norm.detach().cpu().tolist()
    h_vi_norm = [round(elem, 4) for elem in h_vi_norm]
    h_g = round(h_g, 4)
    h_g_norm = round(h_g_norm, 4)

    del _b_vi, edge_index
    gc.collect()
    torch.cuda.empty_cache()
                    
    return h_g_norm, h_vi_norm

