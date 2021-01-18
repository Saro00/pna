import torch
import dgl



if __name__ == "__main__":
    g1 = dgl.DGLGraph()
    g1.add_nodes(2)                                # Add 2 nodes
    g1.add_edge(0, 1)                              # Add edge 0 -> 1
    g1.ndata['hv'] = torch.tensor([[0.], [1.]])       # Initialize node features
    g1.edata['he'] = torch.tensor([[0.]])             # Initialize edge features
    g2 = dgl.DGLGraph()
    g2.add_nodes(3)                                # Add 3 nodes
    g2.add_edges([0, 2], [1, 1])                   # Add edges 0 -> 1, 2 -> 1
    g2.ndata['hv'] = torch.tensor([[2.], [3.], [4.]]) # Initialize node features
    g2.edata['he'] = torch.tensor([[1.], [2.]])       # Initialize edge features
    bg = dgl.batch([g1, g2], edge_attrs=None)
    

    print(bg)
    