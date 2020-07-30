import torch.nn as nn
import dgl
from nets.gru import GRU
from nets.eig_layer import EIGLayer
from nets.mlp_readout_layer import MLPReadout




class EIGNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']

        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.NN_eig = net_params['NN_eig']
        self.avg_d = net_params['avg_d']
        self.towers = net_params['towers']
        self.divide_input_first = net_params['divide_input_first']
        self.divide_input_last = net_params['divide_input_last']
        self.residual = net_params['residual']

        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([EIGLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                                              graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                              residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                              avg_d=self.avg_d, towers=self.towers, edge_features=self.edge_feat, NN_eig = self.NN_eig,
                                              edge_dim=edge_dim, divide_input=self.divide_input_first,
                                              pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers) for _
                                     in range(n_layers - 1)])
        self.layers.append(EIGLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, towers=self.towers, edge_features=self.edge_feat,
                                    NN_eig=self.NN_eig,
                                    edge_dim=edge_dim, divide_input=self.divide_input_last,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers))

        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)

        self.MLP_layer = MLPReadout(2*out_dim, 1)      # 1 out dim since regression problem

    def reset_params(self):
        for layer in self.layers:
            layer.reset_params()

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

        g.ndata['h'] = h

        return h

    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)

        return torch.sigmoid(x)

    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss

        return loss