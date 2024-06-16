import copy
from models import *

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


class Trainer(object):

    def __init__(self, args, graph, train_nodes, val_nodes, test_nodes):
        
        self.args = args
        self.graph = graph

        self.device = torch.device(self.args.device)
        torch.cuda.set_device(self.device)

        self.in_channels = self.graph.x.size(1)
        self.hid_channels = self.args.hid_dim
        self.out_channels = int(torch.max(self.graph.y).item() + 1)

        self.train_nodes, self.val_nodes, self.test_nodes = train_nodes, val_nodes, test_nodes
        
        loss_weight = None
        self.criterion = nn.NLLLoss(weight=loss_weight)

    def model_init(self):

        if self.args.model == 'simple-gnn': Model = Simplified_GNN
        if self.args.model == 'simple-gnn-sym': Model = Simplified_GNN_Sym
        
        if self.args.model == 'gcn': Model = GCN_Model
        if self.args.model == 'gcn2': Model = GCNII_Model
        if self.args.model == 'aero': Model = AERO_GNN_Model
        if self.args.model == 'gpr': Model = GPR_GNN_Model
        
        model = Model(self.args,
                        self.in_channels,
                        self.hid_channels,
                        self.out_channels,
                        self.graph,
                        )
    
        model = model.to(self.device)

        return model
        
    def score(self, graph, model, index_set):

        model.eval()
        with torch.no_grad():
            prediction = model(graph.x, graph.edge_index)
            logits = F.log_softmax(prediction, dim=1)
            val_loss = self.criterion(logits[index_set], graph.y[index_set])

            _, pred = logits.max(dim=1)
            true_false = pred[index_set].eq(graph.y[index_set])
            correct = true_false.sum().item()
            acc = correct / len(index_set)

            return acc, val_loss, true_false, prediction

    def fit(self, graph, model):
        optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=self.args.lr, 
                                        weight_decay=self.args.dr)
            
        iterator = trange(self.args.epochs, desc='Val loss: ', leave=False)

        step_counter = 0
        self.best_val_acc = 0
        self.best_val_loss = np.inf

        for _ in iterator:
            
            model.train()
            optimizer.zero_grad()

            prediction = model(graph.x, graph.edge_index)
            prediction = F.log_softmax(prediction, dim=1)

            loss = F.nll_loss(prediction[self.train_nodes], graph.y[self.train_nodes])
            loss.backward()
            optimizer.step()

            val_acc, val_loss, val_corr, val_logits = self.score(graph, model, self.val_nodes)
            iterator.set_description("Val Loss: {:.4f}".format(val_loss))

            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                step_counter = 0
                
            else:
                step_counter = step_counter + 1
                if step_counter > self.args.patience:    
                    iterator.close()
                    break

        return best_model

    def eval(self, graph, best_model):

        train_acc, train_loss, train_corr, train_logits = self.score(graph, best_model, self.train_nodes)
        val_acc, val_loss, val_corr, val_logits = self.score(graph, best_model, self.val_nodes)
        test_acc, test_loss, test_corr, test_logits = self.score(graph, best_model, self.test_nodes)

        return train_acc, val_acc, test_acc, test_corr
    
    def pseudo_shuffle(self, graph, best_model):

        best_model.eval()
        y_one_hot = F.one_hot(graph.y, self.out_channels).float()
        num_classes = y_one_hot.size(1)

        prediction = best_model(graph.x, graph.edge_index)
        probs = F.softmax(prediction, dim=1)

        low_thr = 0.7
        high_thr = 1
        mask = (probs > low_thr) & (probs < high_thr)
        preds = torch.zeros_like(probs)
        preds[mask] = 1

        preds[self.train_nodes] = y_one_hot[self.train_nodes]
        preds[self.val_nodes] = y_one_hot[self.val_nodes]
        
        for ell in range(num_classes):

            c_preds = preds[:, ell]
            c_idx = (c_preds == 1).nonzero(as_tuple=False).view(-1).to('cpu')

            permute_idx = torch.randperm(c_idx.shape[0])
            graph.x[c_idx] = graph.x[c_idx][permute_idx]

        return graph