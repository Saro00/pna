EPS = 1e-5
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators import AGGREGATORS
from .layers import MLP, FCLayer
from .scalers import SCALERS


class EIGLayerSimple(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, residual, avg_d,
                 posttrans_layers=1):
        super().__init__()
        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.aggregators = aggregators
        self.scalers = scalers

        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers)) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        return {'e': edges.src['h']}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        D = h.shape[-2]
        to_cat = []
        for aggregate in self.aggregators:
            to_cat.append(aggregate(self, h))

        h = torch.cat(to_cat, dim=1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):

        print(h.shape)
        print(h)

        h_in = h
        g.ndata['h'] = h

        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        # Fix shape
        h_new = []
        for l in h:
            m = []
            for _ in range(10):
                m.extend(l)
            h_new.append(m[:75])
        h = torch.cuda.FloatTensor([list(x) for x in h_new])

        print(h.shape)
        print(h)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)

        return h



class EIGLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, type_net, residual, towers=5, divide_input=True,
                 edge_features=None, edge_dim=None, pretrans_layers=1, posttrans_layers=1,):
        super().__init__()
        self.type_net = type_net

        if type_net == 'simple':
            self.model = EIGLayerSimple(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm, batch_norm=batch_norm, residual=residual,
                                   aggregators=aggregators, scalers=scalers, avg_d=avg_d, posttrans_layers=posttrans_layers)

def __repr__(self):
    return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)