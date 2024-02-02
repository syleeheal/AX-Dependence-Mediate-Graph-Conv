import torch

def feature_shuffle(args, shuffled_node_ratio, graph, label, train_nodes, val_nodes, test_nodes):
    """
    feature shuffle for each class
    """
    graph = graph.clone()
    num_class = int(label.max() + 1)

    for c in range(num_class):
        c_idx = (label == c).nonzero(as_tuple=False).view(-1).to('cpu')

        if train_nodes is not None:
            c_idx_train = c_idx[torch.where(torch.isin(c_idx, train_nodes))[0]]
            c_idx_val = c_idx[torch.where(torch.isin(c_idx, val_nodes))[0]]
            c_idx_test = c_idx[torch.where(torch.isin(c_idx, test_nodes))[0]]
            c_idx_rest = c_idx[torch.where(~torch.isin(c_idx, torch.cat([train_nodes, val_nodes, test_nodes])))[0]]

            c_idx_train = c_idx_train[torch.randperm(c_idx_train.shape[0])][:int(c_idx_train.shape[0] * shuffled_node_ratio)]
            c_idx_val = c_idx_val[torch.randperm(c_idx_val.shape[0])][:int(c_idx_val.shape[0] * shuffled_node_ratio)]
            c_idx_test = c_idx_test[torch.randperm(c_idx_test.shape[0])][:int(c_idx_test.shape[0] * shuffled_node_ratio)]
            c_idx_rest = c_idx_rest[torch.randperm(c_idx_rest.shape[0])][:int(c_idx_rest.shape[0] * shuffled_node_ratio)]
            
            graph.x[c_idx_train] = graph.x[c_idx_train][torch.randperm(c_idx_train.shape[0])]
            graph.x[c_idx_val] = graph.x[c_idx_val][torch.randperm(c_idx_val.shape[0])]
            graph.x[c_idx_test] = graph.x[c_idx_test][torch.randperm(c_idx_test.shape[0])]
            graph.x[c_idx_rest] = graph.x[c_idx_rest][torch.randperm(c_idx_rest.shape[0])]


        else:
            c_idx = c_idx[torch.randperm(c_idx.shape[0])][:int(c_idx.shape[0] * shuffled_node_ratio)]
            graph.x[c_idx] = graph.x[c_idx][torch.randperm(c_idx.shape[0])]

    return graph

def feature_noise(args, noised_node_ratio, graph):
    """
    feature noisying by shuffling regardless of class
    """
    graph = graph.clone()

    idx = torch.arange(graph.x.shape[0])
    idx = idx[torch.randperm(idx.shape[0])][:int(idx.shape[0] * noised_node_ratio)]
    graph.x[idx] = graph.x[idx][torch.randperm(idx.shape[0])]

    return graph
