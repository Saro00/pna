import threading
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators import AGGREGATORS, aggregate_NN
from .layers import MLP, FCLayer
from .scalers import SCALERS

"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class EIGLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d,
                 pretrans_layers, posttrans_layers, NN_eig=False, edge_features=False, edge_dim=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features
        self.NN_eig = NN_eig

        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_features)

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        self.scalers = [SCALERS[scale] for scale in scalers.split()]
        self.pretrans = MLP(in_size=2 * in_features + (edge_dim if edge_features else 0), hidden_size=in_features,
                            out_size=in_features, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        if NN_eig:
            self.posttrans = MLP(in_size=((len(aggregators.split())+1) * len(scalers.split()) + 1) * in_features,
                             hidden_size=out_features,
                             out_size=out_features, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        else:
            self.posttrans = MLP(in_size=(len(aggregators.split()) * len(scalers.split()) + 1) * in_features,
                             hidden_size=out_features,
                             out_size=out_features, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.eigfilt1 = MLP(in_size=2, hidden_size=7, out_size=1, layers=3, mid_activation='relu', last_activation='none')
        self.eigfilt2 = MLP(in_size=2, hidden_size=7, out_size=1, layers=3, mid_activation='relu', last_activation='none')
        self.eigfilt = MLP(in_size=6,  out_size=1, layers=2, last_activation='relu')
        self.avg_d = avg_d

    def pretrans_edges(self, edges):

        if self.edge_features:
            z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)

        return {'e': self.pretrans(z2), 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):

        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'], 'eig_d': edges.data['eig_d']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]
        if self.NN_eig:
            #w1 = self.eigfilt1(torch.cat([eig_s[:, :, 1].unsqueeze(-1), eig_d[:][:, :, 1].unsqueeze(-1)], dim=-1))
            #w2 = self.eigfilt2(torch.cat([eig_s[:, :, 2].unsqueeze(-1), eig_d[:][:, :, 2].unsqueeze(-1)], dim=-1))
            w = self.eigfilt(torch.cat([eig_s[:, :, i].unsqueeze(-1) for i in range(1, 4)] +
                                       [eig_d[:][:, :, i].unsqueeze(-1) for i in range(1, 4)], dim=-1))
            w_norm = w / torch.sum(w, dim=1, keepdim=True)
            #e1 = aggregate_NN(h, w1)
            #e2 = aggregate_NN(h, w2)
            e = aggregate_NN(h, w_norm)

        h = torch.cat([aggregate(h, eig_s, eig_d) for aggregate in self.aggregators], dim=1)

        if self.NN_eig:
            #h = torch.cat([h, e1, e2], dim=1)
            h = torch.cat([h, e], dim=1)

        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        g.ndata['h'] = h

        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata['ef'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_features, self.out_features)
