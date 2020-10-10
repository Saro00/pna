import time
import dgl
import torch
from torch.utils.data import Dataset
import random as rd
from ogb.linkproppred import DglLinkPropPredDataset, LinkPropPredDataset, Evaluator


from scipy import sparse as sp
import numpy as np
import networkx as nx


def positional_encoding(g, pos_enc_dim, norm):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    if norm == 'none':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1), dtype=float)
        L = N * sp.eye(g.number_of_nodes()) - A
    elif norm == 'sym':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N
    elif norm == 'walk':
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata['eig'] = torch.from_numpy(np.real(EigVec[:, :pos_enc_dim + 1])).float()

    return g


class COLLABDataset(Dataset):
    def __init__(self, name, norm='none', verbose=True):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglLinkPropPredDataset(name='ogbl-collab')

        self.graph = self.dataset[0]  # single DGL graph
        #self._add_positional_encodings(10, norm)
        self._add_eig(norm=norm, number=6)

        # Create edge feat by concatenating weight and year
        self.graph.edata['feat'] = torch.cat(
            [self.graph.edata['edge_weight'], self.graph.edata['edge_year']],
            dim=1
        )

        self.split_edge = self.dataset.get_edge_split()
        self.train_edges = self.split_edge['train']['edge']  # positive train edges
        self.val_edges = self.split_edge['valid']['edge']  # positive val edges
        self.val_edges_neg = self.split_edge['valid']['edge_neg']  # negative val edges
        self.test_edges = self.split_edge['test']['edge']  # positive test edges
        self.test_edges_neg = self.split_edge['test']['edge_neg']  # negative test edges

        self.evaluator = Evaluator(name='ogbl-collab')
        if verbose:
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def _add_positional_encodings(self, pos_enc_dim, norm):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph = positional_encoding(self.graph, pos_enc_dim, norm)

    def _add_eig(self, norm='none', number=6):

        dataset = LinkPropPredDataset(name='ogbl-collab')
        graph = dataset[0]
        G = nx.Graph()
        G.add_nodes_from([i for i in range(235868)])

        for nod1, nod2 in zip(graph['edge_index'][0], graph['edge_index'][1]):
            G.add_edge(nod1, nod2)

        components = list(nx.connected_components(G))
        list_G = []
        list_nodes = []

        for component in components:
            G_new = nx.Graph()
            G_new.add_nodes_from(list(component))
            list_G.append(G_new)
            list_nodes.append(list(component))
        for i in range(len(list_G)):
            for nod1, nod2 in list(G.edges(list_nodes[i])):
                list_G[i].add_edge(nod1, nod2)

        EigVec_global = np.ones((235868, number))
        for g in list_G:
            node_list = list(g.nodes)
            A = nx.adjacency_matrix(g, nodelist=node_list).astype(float)
            N = sp.diags(list(map(lambda x: x[1], g.degree())))
            if norm == 'none':
                D = N * sp.eye(len(node_list))
                L = D - A
            elif norm == 'sym':
                D = N * sp.eye(len(node_list))
                L = D ** (-1 / 2) * (D - A) * D ** (-1 / 2)
            elif norm == 'walk':
                D = N * sp.eye(len(node_list))
                L = D ** (-1) * (D - A)

            if len(node_list) > 2:
                EigVal, EigVec = sp.linalg.eigs(L, k=min(len(node_list) - 2, number), which='SR', tol=1e-2)
                EigVec = EigVec[:, EigVal.argsort()]
                EigVec_global[node_list, : min(len(node_list) - 2, number)] = EigVec[:, :]
            elif len(node_list) == 2:
                EigVec_global[node_list[0], :number] = np.zeros((1, number))
        self.graph.ndata['eig'] = torch.from_numpy(EigVec_global).float()