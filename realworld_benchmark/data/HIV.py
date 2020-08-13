import time
import dgl
import torch
from torch.utils.data import Dataset
import random as rd
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.graphproppred import Evaluator


from scipy import sparse as sp
import numpy as np
import itertools
import torch.utils.data



def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata['eig'] = torch.from_numpy(np.real(EigVec[:, 1:pos_enc_dim + 1])).float()

    return g


class HIVDGL(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = [g[0] for g in self.data]
        self.graph_labels = [g[1] for g in self.data]
        self.n_samples = len(self.data)
        self.get_eig()


    def get_eig(self):
        self.graph_lists = [positional_encoding(g, 7) for g in self.graph_lists]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class HIVDataset(Dataset):
    def __init__(self, name, verbose=True):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglGraphPropPredDataset(name = 'ogbg-molhiv')
        self.split_idx = self.dataset.get_idx_split()
        self.train = HIVDGL(self.dataset, self.split_idx['train'])
        self.val = HIVDGL(self.dataset, self.split_idx['valid'])
        self.test = HIVDGL(self.dataset, self.split_idx['test'])

        self.evaluator = Evaluator(name='ogbg-molhiv')

        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]