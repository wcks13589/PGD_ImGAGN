import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import Pyg2Dpr

from torch_geometric.datasets import Planetoid


class GraphData:
    def __init__(self, dataset, sparse=False, normalize=True):
        data = Pyg2Dpr(Planetoid('../data', name=dataset))
        
        if sparse:
            self.adj = self.sparse_mx_to_torch_sparse_tensor(data.adj)
        else:
            self.adj = torch.FloatTensor(data.adj.todense())
            
        if normalize:
            self.x = torch.FloatTensor(self.normalize(data.features))
        else:
            self.x = torch.FloatTensor(data.features)
            
            
        self.y = torch.LongTensor(data.labels)
        
        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test
        
        self.edge_index = torch.stack(self.adj.nonzero(as_tuple=True))
        self.edge_attr = self.adj[self.edge_index[0], self.edge_index[1]]
        
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow, sparsecol),1)
        sparsedata=torch.FloatTensor(sparse_mx.data)
        return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))
    
    def load_feature_graph(self):
        
        x_feat = self.create_nodes()
        edge_index_feat, edge_attr_feat = self.create_edges()
        
        return x_feat, edge_index_feat, edge_attr_feat

    def create_edges(self):
        n_samples = self.x.shape[0]
        
        edge_index = torch.stack(self.x.nonzero(as_tuple=True))
        edge_attr = self.x[edge_index[0], edge_index[1]]

        # to undirected
        edge_index[1] += n_samples
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, edge_index_reverse], -1)
        edge_attr = torch.cat([edge_attr, edge_attr])
        
        return edge_index, edge_attr
    
    def create_nodes(self, sample_onehot=False):
        
        n_row, n_col = self.x.shape
        
        if sample_onehot:
            sample_nodes = torch.zeros(n_row, n_col+1)
            sample_nodes[:,0] = 1
            feature_nodes = torch.cat([torch.zeros(n_col).unsqueeze(1), torch.eye(n_col)], -1)
            
        else:
            sample_nodes = torch.ones(n_row, n_col)
            feature_nodes = torch.eye(n_col)
            
        return torch.cat([sample_nodes, feature_nodes])
    
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


        
    