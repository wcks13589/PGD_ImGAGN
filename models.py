import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch

class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, generate_node, min_node):
        super(GCN, self).__init__()


        self.nfeat = nfeat
        self.hidden_sizes = nhid
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nhid, 2)
        self.attention = Attention(nfeat*2, 1)
        self.generate_node = generate_node
        self.min_node = min_node
        self.dropout = dropout
        self.eps = 1e-10

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.gc2(x, adj)
        x2 = self.gc3(x, adj)

        return F.log_softmax(x1, dim=1), F.log_softmax(x2, dim=1), F.softmax(x1, dim=1)[:,-1]

    def get_embedding(self,x , adj):
        x = F.relu(self.gc1(x, adj))
        x = torch.spmm(adj, x)
        return x

class Generator(nn.Module):
    def __init__(self,  dim):
        super(Generator, self).__init__( )

        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, dim)
        self.fc4 = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = (x+1)/2
        return x

    def generate_fake_features(self, adj_min, features, minority, fake_nodes):
        n_nodes = len(features) + len(adj_min)
        n_minority = len(minority)
        
        adj_min_norm = F.softmax(adj_min[:,:n_minority], dim=1)
        gen_imgs = torch.mm(adj_min_norm, features[minority])

        matr = F.softmax(adj_min[:,:n_minority], dim =1).data.cpu().numpy()
        pos = np.where(matr>1/matr.shape[1])
        adj_temp = sp.coo_matrix((np.ones(pos[0].shape[0]),
                                  (fake_nodes[pos[0]].numpy(), minority[pos[1]].numpy())),
                                  shape=(n_nodes, n_nodes),
                                  dtype=np.float32)

        return gen_imgs, adj_temp



