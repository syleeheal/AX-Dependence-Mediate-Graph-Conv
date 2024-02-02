import torch
from torch_geometric.data import Data


def csbmx_graph(args):
    
    """
    initialize Y by assigning half 0 and half 1
    """
    y = torch.zeros((args.num_nodes, ), dtype=torch.long)
    y[args.num_nodes//2:] = 1
    

    """
    sample X
    """
    if args.k == 1:
        x0_sampler = torch.distributions.normal.Normal(0, 1)
        x1_sampler = torch.distributions.normal.Normal(0, args.sigma)
    elif args.k > 1:
        x0_sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.k), torch.eye(args.k))
        x1_sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.k), torch.eye(args.k) * args.sigma)
        
    x0 = x0_sampler.sample((y[y==0].shape[0], 1)).to(args.device)
    x1 = x1_sampler.sample((y[y==1].shape[0], 1)).to(args.device)
    if args.k > 1:
        x0 = x0.squeeze()
        x1 = x1.squeeze()

    """
    h_p_ij to edge sampling prob
    """
    delta_x0 = torch.cdist(x0, x0, p=2)
    delta_x1 = torch.cdist(x1, x1, p=2)
    delta_x01 = torch.cdist(x0, x1, p=2)
    delta_x10 = torch.cdist(x1, x0, p=2)

    h_p_ij_x0 = (delta_x0.sum(dim=1, keepdim=True) / (delta_x0.shape[0] - 1)) - delta_x0
    h_p_ij_x1 = (delta_x1.sum(dim=1, keepdim=True) / (delta_x1.shape[0] - 1)) - delta_x1
    h_p_ij_x01 = delta_x01.mean(dim=1, keepdim=True) - delta_x01 
    h_p_ij_x10 = delta_x10.mean(dim=1, keepdim=True) - delta_x10

    x0_prob = tempered_exp(args.tau, h_p_ij_x0, contains_self=True)
    x1_prob = tempered_exp(args.tau, h_p_ij_x1, contains_self=True)
    x01_prob = tempered_exp(args.tau, h_p_ij_x01, contains_self=False)
    x10_prob = tempered_exp(args.tau, h_p_ij_x10, contains_self=False)

    for i in range(x0_prob.shape[0]): x0_prob[i, i] = 0  # prob = 0 for self-loop
    for i in range(x1_prob.shape[0]): x1_prob[i, i] = 0  # prob = 0 for self-loop
        

    """
    weighted sampling of directed edges without replacement
    """
    idx_0 = torch.multinomial(x0_prob, args.d_pos, replacement=False)
    idx_1 = torch.multinomial(x1_prob, args.d_pos, replacement=False)
    idx_01 = torch.multinomial(x01_prob, args.d_neg, replacement=False)
    idx_10 = torch.multinomial(x10_prob, args.d_neg, replacement=False)

    edge_0 = torch.zeros_like(x0_prob)
    edge_1 = torch.zeros_like(x1_prob)
    edge_01 = torch.zeros_like(x01_prob)
    edge_10 = torch.zeros_like(x10_prob)

    for i in range(x0_prob.shape[0]): edge_0[i, idx_0[i]] = 1
    for i in range(x1_prob.shape[0]): edge_1[i, idx_1[i]] = 1
    for i in range(x01_prob.shape[0]): edge_01[i, idx_01[i]] = 1
    for i in range(x10_prob.shape[0]): edge_10[i, idx_10[i]] = 1

    edge = torch.cat([torch.cat([edge_0, edge_01], dim=1),
                      torch.cat([edge_10, edge_1], dim=1)], dim=0)

    edge_index = edge.nonzero(as_tuple=False).t()


    """
    non-class-controlled features
    """
    x0 = (x0 - args.mu)
    x1 = (x1 + args.mu)
    x = torch.cat([x0, x1], dim=0)


    """
    to pyg graph
    """
    graph = Data(x=x, y=y, edge_index=edge_index)

    return graph


def tempered_exp(t, x, contains_self):
    
    if t > 0 and contains_self:
        x_max = torch.topk(x, 2, dim=1)[0][:, 1].view(-1, 1)
        x = x - x_max
    
    elif t > 0 and not contains_self:
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x = x - x_max

    if t < 0:
        x_min = torch.topk(x, 2, dim=1, largest=False)[0][:, 1].view(-1, 1)
        x = x - x_min

    return torch.exp(t * x)

