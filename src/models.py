import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GCN2Conv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops



class GCN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)
        self.num_edges = graph.edge_index.size(1)
        
        self.args.num_layers = int(self.args.num_layers)

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):

        self.convs = nn.ModuleList()
        inchannels = self.in_channels

        for i in range(self.args.num_layers-1):
            self.convs.append(GCNConv(inchannels, self.hid_channels, cached=True, normalize=True, add_self_loops=False))
            inchannels = self.hid_channels

        self.convs.append(GCNConv(inchannels, self.out_channels, cached=True, normalize=True, add_self_loops=False))

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = F.relu

    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()

    def forward(self, x, edge_index):
        
        x = self.dropout(x)

        for i in range(self.args.num_layers-1):
            x = self.convs[i](x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)

        return x

class Simplified_GNN(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.setup_layers()

    def setup_layers(self):

        self.linear = Linear(self.in_channels, self.out_channels, bias=True, weight_initializer='glorot')


    def forward(self, x, edge_index):
        degree = scatter_add(torch.ones_like(edge_index[0]), edge_index[0])
        edge_weight = 1 / degree[edge_index[1]]

        ax = x
        for _ in range(self.args.num_layers):
            ax = self.propagate(edge_index, x=ax, edge_weight=edge_weight)
        axw = self.linear(ax)

        return axw

class Simplified_GNN_Sym(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.setup_layers()

    def setup_layers(self):

        self.linear = Linear(self.in_channels, self.out_channels, bias=True, weight_initializer='glorot')


    def forward(self, x, edge_index):

        edge_index, edge_weight = gcn_norm(edge_index, num_nodes = x.shape[0], add_self_loops=False)
        ax = x
        for _ in range(self.args.num_layers):
            ax = self.propagate(edge_index, x=ax, edge_weight=edge_weight)
        axw = self.linear(ax)

        return axw

class GPR_GNN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super(GPR_GNN_Model, self).__init__(aggr='add')

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.K = int(self.args.iterations)
        self.alpha = self.args.alpha

        TEMP = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        TEMP[-1] = (1-self.alpha)**self.K
        self.temp = nn.Parameter(torch.tensor(TEMP))

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = nn.ReLU()
        
        self.linear_node_1 = Linear(self.in_channels, self.hid_channels, bias=True, weight_initializer='glorot')
        self.linear_node_2 = Linear(self.hid_channels, self.out_channels, bias=True,weight_initializer='glorot')


    def reset_parameters(self):

        self.linear_node_1.reset_parameters()
        self.linear_node_2.reset_parameters()

        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def node_label_pred(self, x):
        
        x = self.dropout(x)
        
        h = self.linear_node_1(x)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.linear_node_2(h)

        return h

    def gpr_propagate(self, a, h, edge_idx):
        
        z = h*(self.temp[0])

        for k in range(self.K):

            h = self.propagate(edge_idx, x=h, norm = a)

            gamma = self.temp[k+1]
            z = z + gamma*h

        return z
        
    def forward(self, x, edge_idx):
        
        h = self.node_label_pred(x)
        edge_idx, a = gcn_norm(edge_idx, num_nodes = self.num_nodes, add_self_loops=False)
        z = self.gpr_propagate(a, h, edge_idx)
        
        return z

    def message(self, x_j, norm):
        return x_j * norm.view(-1, 1)

class AERO_GNN_Model(MessagePassing):

    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__(node_dim=0, )

        self.args = args

        self.num_nodes = graph.x.size(0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = 1
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = int(self.args.iterations)
        self.args.num_layers = int(self.args.num_layers)
                
        self.setup_layers()
        self.reset_parameters()


    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.atts = nn.ParameterList()
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []

        self.dense_lins.append(Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'))
        for _ in range(self.args.num_layers - 1): self.dense_lins.append(Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'))
        self.dense_lins.append(Linear(self.hid_channels_, self.out_channels, bias=True, weight_initializer='glorot'))

        for k in range(self.K + 1): 
            self.atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels)))
            self.hop_atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels*2)))
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, self.heads)))
            self.decay_weights.append( np.log((self.args.lambd / (k+1)) + (1 + 1e-6)) )
        self.hop_atts[0]=nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels))
        self.atts = self.atts[1:]


    def reset_parameters(self):
        
        for lin in self.dense_lins: lin.reset_parameters()
        for att in self.atts: glorot(att) 
        for att in self.hop_atts: glorot(att) 
        for bias in self.hop_biases: ones(bias) 


    def hid_feat_init(self, x):
        
        x = self.dropout(x)
        x = self.dense_lins[0](x)

        for l in range(self.args.num_layers - 1):
            x = self.elu(x)
            x = self.dropout(x)
            x = self.dense_lins[l+1](x)
        
        return x


    def aero_propagate(self, h, edge_index):
        
        self.k = 0
        h = h.view(-1, self.heads, self.hid_channels)
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[self.k]

        for k in range(self.K):

            self.k = k+1
            h = self.propagate(edge_index, x = h, z_scale = z_scale)            
            g = self.hop_att_pred(h, z_scale)
            z += h * g
            z_scale = z * self.decay_weights[self.k]
                
        return z


    def node_classifier(self, z):
        
        z = z.view(-1, self.heads * self.hid_channels)
        z = self.elu(z)
        z = self.dense_lins[-1](z)
        
        return z


    def forward(self, x, edge_index):
        
        h0 = self.hid_feat_init(x)
        z_k_max = self.aero_propagate(h0, edge_index)
        z_star =  self.node_classifier(z_k_max)
        return z_star


    def hop_att_pred(self, h, z_scale):

        if z_scale is None: 
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, self.heads, int(x.shape[-1]))
        g = self.elu(g)
        g = (self.hop_atts[self.k] * g).sum(dim=-1) + self.hop_biases[self.k]
        
        return g.unsqueeze(-1)


    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index):
        
        # edge attention (alpha_check_ij)
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.atts[self.k-1] * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(a_ij, col, dim=0, dim_size=self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]        

        return a_ij


    def message(self, edge_index, x_j, z_scale_i, z_scale_j):
        a = self.edge_att_pred(z_scale_i, z_scale_j, edge_index)
        return a.unsqueeze(-1) * x_j

class GCNII_Model(MessagePassing):
    
    def __init__(self, args, in_channels, hid_channels, out_channels, graph,):
        super().__init__()

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.num_nodes = graph.x.size(0)

        self.setup_hyperparameters()
        self.setup_layers()

    def setup_hyperparameters(self):
        self.alpha = self.args.alpha
        self.theta = self.args.lambd
        self.num_layer = int(self.args.num_layers)

    def setup_layers(self):

        self.dropout = nn.Dropout(self.args.dropout)
        self.relu = nn.ReLU()

        self.lins = nn.ModuleList()
        self.lins.append(Linear(self.in_channels, self.hid_channels))
        self.lins.append(Linear(self.hid_channels, self.out_channels))
        
        self.convs = nn.ModuleList()

        for layer in range(self.num_layer):
            self.convs.append(
                                GCN2Conv(channels = self.hid_channels,
                                alpha = self.alpha,
                                theta = self.theta,
                                layer = layer + 1,
                                shared_weights = True,
                                normalize = False,
                                )
            )

    def forward(self, x, edge_index):

        edge_index, edge_weight = gcn_norm(edge_index, num_nodes = self.num_nodes, add_self_loops=False, dtype=x.dtype)
        
        x = self.dropout(x)
        x = x_0 = self.lins[0](x).relu()
        
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x, x_0, edge_index, edge_weight)
            x = x.relu()

        x = self.lins[1](x)

        return x

class Propagation_Layer(MessagePassing):

    def __init__(self, K: int = 1,add_self_loops: bool = False, **kwargs):
        super(Propagation_Layer, self).__init__(aggr='add')

        self.K = K
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):
        
        # add self loops in the edge
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        degree = scatter_add(torch.ones_like(edge_index[0]), edge_index[0])
        edge_weight = 1 / degree[edge_index[1]]

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, num_nodes=x.size(0))

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
