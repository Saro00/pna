import torch.nn as nn
import dgl
from nets.gru import GRU
from nets.eig_layer import EIGLayer, VirtualNode
from nets.mlp_readout_layer import MLPReadout
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder






class EIGNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        decreasing_dim = net_params['decreasing_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.JK = net_params['JK']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.gru_enable = net_params['gru']
        device = net_params['device']
        self.virtual_node = net_params['virtual_node']

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)

        if self.edge_feat:
            self.embedding_e = BondEncoder(emb_dim=edge_dim)

        self.layers = nn.ModuleList([EIGLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                      edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
             in range(n_layers - 1)])
        self.layers.append(EIGLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                                    edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)
        if self.gru_enable:
            self.gru = GRU(hidden_dim, hidden_dim, device)

        self.MLP_layer = MLPReadout(out_dim, 128, decreasing_dim=decreasing_dim)

        self.virtual_node_layers = None
        if (self.virtual_node is not None) and (self.virtual_node.lower() != 'none'):
            self.virtual_node_layers = \
                nn.ModuleList([
                VirtualNode(dim=hidden_dim, dropout=dropout, batch_norm=self.batch_norm,
                            bias=True, vn_type=self.virtual_node)
                for _ in range(n_layers - 1)])



    def forward(self, g, h, e, snorm_n, snorm_e):

        #
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.JK == 'sum':
            h_list = [h]
        if self.edge_feat:
            e = self.embedding_e(e)

        # Loop all layers
        for i, conv in enumerate(self.layers):
            # Graph conv layers
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

            # Virtual node layer
            if self.virtual_node_layers is not None:
                if i == 0:
                    vn_h = 0
                vn_h, h = self.virtual_node_layers[i].forward(g, h, vn_h)

            # Append list of features for jumping knowledge
            if self.JK == 'sum':
                h_list.append(h)

        g.ndata['h'] = h
        
        if self.JK == 'last': # Take the last layer as the conv readout
            pass

        elif self.JK == 'sum': # Jumping knowledge (summing all layers outputs)
            h = 0
            for layer in h_list:
                h += layer
            g.ndata['h'] = h

        # Readout layer
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        loss = torch.nn.BCEWithLogitsLoss()(scores, labels)

        return loss