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


def csbmx_two_graph(args):
    """
    initialize Y by assigning evenly to 10 classes
    """
    y = torch.zeros((args.num_nodes, ), dtype=torch.long).to(args.device)
    num_class = 10
    for ell in range(num_class):
        y[args.num_nodes//num_class*ell : args.num_nodes // num_class*(ell+1)] = ell
    

    """
    sample X
    """
    x_sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(num_class), torch.eye(num_class))
    
    x_list = []
    for ell in range(num_class):
        x_list.append(x_sampler.sample((y[y==ell].shape[0], 1)).to(args.device).squeeze())
    x = torch.cat(x_list, dim=0)

    """
    h_p_ij to edge sampling prob
    """
    delta_x = torch.cdist(x, x, p=2)
    h_p_ij = delta_x.mean(dim=1, keepdim=True) - delta_x
    x_prob = tempered_exp(args.tau, h_p_ij, contains_self=True)
    for i in range(x_prob.shape[0]): x_prob[i, i] = 0  # prob = 0 for self-loop
    

    """
    sample power-law degree distribution
    """
    degrees = generate_power_law_samples(alpha=1.5, num_samples=args.num_nodes, xmin=10, xmax=1000)
    degrees = degrees.int()
    

    """
    weighted sampling of directed edges without replacement
    """
    edge_idx_list = []
    pos_ratio = args.d_pos / (args.d_pos + args.d_neg)
    for node_i in (range(args.num_nodes)):
        pos_degree = int(degrees[node_i] * pos_ratio)
        neg_degree = degrees[node_i] - pos_degree
        
        y_i = y[node_i]
        
        x_prob_i = x_prob[node_i]
        x_prob_i_intra_class = x_prob_i[y==y_i]
        x_prob_i_inter_class = x_prob_i[y!=y_i]

        idx_pos = torch.multinomial(x_prob_i_intra_class, pos_degree, replacement=False)
        idx_neg = torch.multinomial(x_prob_i_inter_class, neg_degree, replacement=False)
        idx_pos = torch.where(y==y_i)[0][idx_pos]
        idx_neg = torch.where(y!=y_i)[0][idx_neg]
        idx = torch.cat([idx_pos, idx_neg], dim=0)

        edge_idx_list.append(idx)


    edge_empty = torch.zeros_like(x_prob)
    for node_i in range(args.num_nodes):
        edge_empty[node_i, edge_idx_list[node_i]] = 1
    edge_index = edge_empty.nonzero(as_tuple=False).t()


    """
    non-class-controlled features
    """
    for ell in range(num_class):
        x[y==ell, ell] += args.mu
        x[y!=ell, ell] -= args.mu

    """
    to pyg graph
    """
    graph = Data(x=x, y=y.cpu(), edge_index=edge_index)

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



def generate_power_law_samples(alpha, num_samples, xmin=10, xmax=1000):
    """
    Generate samples from a power-law distribution with a minimum value x_min.
    """
    # Sample from a uniform distribution
    uniform_samples = torch.rand(num_samples)

    # Transform the uniform samples to power-law distribution
    power_law_samples = xmin * (uniform_samples ** (1 / (1 - alpha)))
    power_law_samples = torch.clamp(power_law_samples, max=xmax)

    return power_law_samples


