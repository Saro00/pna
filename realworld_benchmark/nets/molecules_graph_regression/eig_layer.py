EPS = 1e-5
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators import AGGREGATORS, aggregate_NN
from .layers import MLP, FCLayer
from .scalers import SCALERS



class EIGTower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, NN_eig, avg_d,
                 pretrans_layers, posttrans_layers, edge_features, edge_dim):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.aggregators = aggregators
        self.scalers = scalers
        self.NN_eig = NN_eig
        self.pretrans = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        if NN_eig:
            self.posttrans = MLP(in_size=((len(aggregators)+1) * len(scalers) + 1) * in_dim,
                             hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        else:
            self.posttrans = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim,
                             hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        self.eigfilt = MLP(in_size=6, hidden_size=3, out_size=1, layers=3, mid_activation='relu', last_activation='Sigmoid')
        self.eigfiltbis = nn.Linear(6, 1, bias=True)
        self.eigfilter = MLP(in_size=3, hidden_size=3, out_size=1, layers=3,  mid_activation='relu', last_activation='none')

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
            w = self.eigfilt(torch.cat([torch.mul(eig_s[:, :, 1:4], torch.sign(eig_s[:, :, 1:4])),
                                        torch.mul(eig_d[:, :, 1:4], torch.sign(eig_s[:, :, 1:4])) ], dim=-1))
            ws = torch.sigmoid(self.eigfilt(torch.cat([eig_s[:, :, 1:4], eig_d[:, :, 1:4]], dim=-1)))
            wb = self.eigfilter(torch.abs(eig_s[:, :, 1:4] - eig_d[:, :, 1:4]))
            wl = self.eigfilt(torch.cat([eig_s[:, :, 1:4], eig_d[:, :, 1:4]], dim=-1))
            w_norm = w / (torch.sum(w, dim=1, keepdim=True) + EPS)
            #e1 = aggregate_NN(h, w1)
            #e2 = aggregate_NN(h, w2)
            e = aggregate_NN(h, ws)
            eb = aggregate_NN(h, wb)
            el = aggregate_NN(h, wl)

        h = torch.cat([aggregate(h, eig_s, eig_d) for aggregate in self.aggregators], dim=1)

        if self.NN_eig:
            #h = torch.cat([h, e1, e2], dim=1)
            h = torch.cat([h, el], dim=1)

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



class EIGLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, graph_norm, batch_norm, NN_eig=False, towers=1,
                 pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False,
                 edge_dim=0):
        super().__init__()
        assert ((
                    not divide_input) or in_dim % towers == 0), "if divide_input is set the number of towers has to divide in_dim"
        assert (out_dim % towers == 0), "the number of towers has to divide the out_dim"
        assert avg_d is not None

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.NN_eig = NN_eig
        self.edge_features = edge_features
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(EIGTower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, NN_eig=self.NN_eig, avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, batch_norm=batch_norm, dropout=dropout,
                                        graph_norm=graph_norm, edge_features=edge_features, edge_dim=edge_dim))
        # mixing network
        self.mixing_network = FCLayer(out_dim, out_dim, activation='LeakyReLU')

    def forward(self, g, h, e, snorm_n):
        h_in = h  # for residual connection

        if self.divide_input:
            h_cat = torch.cat( [tower(g, h[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower], e, snorm_n)
                 for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h, e, snorm_n) for tower in self.towers], dim=1)

        h_out = self.mixing_network(h_cat)

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out


def __repr__(self):
    return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)