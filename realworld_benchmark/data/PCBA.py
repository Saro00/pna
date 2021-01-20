import time
import dgl
import torch
from torch.utils.data import Dataset
import random as rd
from ogb.graphproppred import Evaluator


from scipy import sparse as sp
import numpy as np
import itertools
import torch.utils.data
import pandas as pd
import shutil, os
import os.path as osp
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_csv_graph_dgl
import networkx as nx

import gc



def positional_encoding(g, pos_enc_dim, norm):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    num = int(g.number_of_nodes())
    G = nx.Graph()
    G.add_nodes_from([i for i in range(num)])
    print(g)
    for nod1, nod2 in zip(g.all_edges()[0].detach(), g.all_edges()[1].detach()):
        G.add_edge(nod1, nod2)

    components = list(nx.connected_components(G))
    list_G = []
    list_nodes = []

    for component in components:
        G_new = nx.Graph()
        G_new.add_nodes_from(list(component))
        list_G.append(G_new)
        list_nodes.append(list(component))
    print('ok')
    for i in range(len(list_G)):
        for nod1, nod2 in list(G.edges(list_nodes[i])):
            list_G[i].add_edge(nod1, nod2)
    print('si')

    EigVec_global = np.ones((num, pos_enc_dim))
    for connected in list_G:
        node_list = list(g.nodes)
        print('here')
        A = nx.adjacency_matrix(connected, nodelist=node_list).astype(float)
        if norm == 'none':
            D = sp.diags(list(map(lambda x: x[1], connected.degree())))
            L = D - A
        elif norm == 'sym':
            D_norm = sp.diags(list(map(lambda x: x[1] ** (-0.5), connected.degree())))
            D = sp.diags(list(map(lambda x: x[1], connected.degree())))
            L = D_norm * (D - A) * D_norm
        elif norm == 'walk':
            D_norm = sp.diags(list(map(lambda x: x[1] ** (-1), connected.degree())))
            D = sp.diags(list(map(lambda x: x[1], connected.degree())))
            L = D_norm * (D - A)
        print('here2')

        if len(node_list) > 2:
            EigVal, EigVec = sp.linalg.eigs(L, k=min(len(node_list) - 2, pos_enc_dim), which='SR', tol=0)
            EigVec = EigVec[:, EigVal.argsort()] / np.max(EigVec[:, EigVal.argsort()], 0)
            EigVec_global[node_list, : min(len(node_list) - 2, pos_enc_dim)] = EigVec[:, :]
        elif len(node_list) == 2:
            EigVec_global[node_list[0], :pos_enc_dim] = np.zeros((1, pos_enc_dim))
    g.ndata['eig'] = torch.from_numpy(EigVec_global).float()
    return g



class DownloadPCBA(object):
    """ Modified version of DglGraphPropPredDataset of ogb.graphproppred, that doesn't save the dataset """

    def __init__(self, name='ogbg-pcba', root="data"):
        self.name = name  ## original name, e.g., ogbg-mol-tox21
        self.dir_name = 'ogbg_molpcba_dgl'
        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.

        self.download_name = 'pcba'  ## name of downloaded file, e.g., tox21

        self.num_tasks = 128
        self.eval_metric = 'ap'
        self.task_type = 'binary classification'
        self.num_classes = 2

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        ### download
        url = 'https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip'
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            # delete folder if there exists
            try:
                shutil.rmtree(self.root)
            except:
                pass
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print("Stop download.")
            exit(-1)

        ### preprocess
        add_inverse_edge = True
        additional_node_files = []
        additional_edge_files = []

        graphs = read_csv_graph_dgl(raw_dir, add_inverse_edge=add_inverse_edge,
                                    additional_node_files=additional_node_files,
                                    additional_edge_files=additional_edge_files)

        labels = pd.read_csv(osp.join(raw_dir, "graph-label.csv.gz"), compression="gzip", header=None).values

        has_nan = np.isnan(labels).any()

        if "classification" in self.task_type:
            if has_nan:
                labels = torch.from_numpy(labels)
            else:
                labels = torch.from_numpy(labels).to(torch.long)
        else:
            labels = torch.from_numpy(labels)

        print('Not Saving...')
        # save_graphs(pre_processed_file_path, graphs, labels={'labels': labels})

        ### load preprocessed files
        self.graphs = graphs
        self.labels = labels

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = 'scaffold'

        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]

        return {"train": torch.tensor(train_idx, dtype=torch.long), "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long)}

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class PCBADGL(torch.utils.data.Dataset):
    def __init__(self, data, split, norm='none'):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = []
        self.graph_labels = []
        for i, g in enumerate(self.data):
            if g[0].number_of_nodes() > 5 and rd.random() < 0.5: # and rd.random() < 0.2:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)
        del self.data


    def get_eig(self, norm):
        self.graph_lists = [positional_encoding(g, 2, norm=norm) for g in self.graph_lists]

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


class PCBADataset(Dataset):
    def __init__(self, name, norm='none', verbose=True):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        dataset = DownloadPCBA(name = 'ogbg-molpcba')
        split_idx = dataset.get_idx_split()
        self.train = PCBADGL(dataset, split_idx['train'], norm=norm)
        self.val = PCBADGL(dataset, split_idx['valid'], norm=norm)
        self.test = PCBADGL(dataset, split_idx['test'], norm=norm)
        del dataset
        del split_idx
        self.train.get_eig(norm=norm)
        self.val.get_eig(norm=norm)
        self.test.get_eig(norm=norm)

        self.evaluator = Evaluator(name='ogbg-molpcba')

        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))

        labels = torch.cat([label.unsqueeze(0) for label in labels])
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]