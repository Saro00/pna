import torch.nn as nn
import torch
import dgl
from nets.gru import GRU
from nets.eig_layer import EIGLayer
from nets.mlp_readout_layer import MLPReadout




class EIGHead(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.NN_eig = net_params['NN_eig']
        self.avg_d = net_params['avg_d']
        self.not_pre = net_params['not_pre']
        self.towers = net_params['towers']
        self.divide_input_first = net_params['divide_input_first']
        self.divide_input_last = net_params['divide_input_last']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, edge_dim)

        self.layers = nn.ModuleList([EIGLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                              graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                              residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                              avg_d=self.avg_d, not_pre=self.not_pre, towers=self.towers, edge_features=self.edge_feat, NN_eig = self.NN_eig,
                                              edge_dim=edge_dim, divide_input=self.divide_input_first,
                                              pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers) for _
                                     in range(n_layers - 1)])
        self.layers.append(EIGLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, not_pre=self.not_pre, towers=self.towers, edge_features=self.edge_feat,
                                    NN_eig=self.NN_eig,
                                    edge_dim=edge_dim, divide_input=self.divide_input_last,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers))

        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)

        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def reset_params(self):
        for layer in self.layers:
            layer.reset_params()

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return hg




class EIGNetSplit(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        self.aggregators = net_params['aggregators']
        self.out_dim = net_params['out_dim']
        net_params_copy = net_params.copy()

        net_params_list = []
        for agg in self.aggregators.split():
            net_params_copy['aggregators'] = agg
            net_params_list.append(net_params_copy)

        self.models = nn.ModuleList([EIGHead(net_params_list[i]) for i in range(len(net_params_list))])
        self.MLP_layer = MLPReadout(self.out_dim * len(self.aggregators.split()), 1)



    def forward(self, g, h, e, snorm_n, snorm_e):
        out = torch.cat([model.forward(g, h, e, snorm_n, snorm_e) for model in self.models], dim=-1)
        return self.MLP_layer(out)


    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss