import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def load_data(dataset="citeseer"):
    path = "./dataset/" + dataset + "/"
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}features.{}".format(path, dataset),
                                        dtype=np.float32)
    features = sp.csr_matrix(idx_features_labels[:, 0:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]

    idx_train = np.genfromtxt("{}train.{}".format(path, dataset),
                              dtype=np.int32).squeeze()

    idx_test = np.genfromtxt("{}test.{}".format(path, dataset),
                             dtype=np.int32).squeeze()

    if len(labels) != len(idx_train) + len(idx_test):
        idx_train, idx_test = train_test_split(range(len(labels)), test_size=0.2)

    if labels.max() > 1:
        minority_count = len(labels)
        minority_class = -1
        for i in np.unique(labels):
            count = sum(labels==i)
            if count < minority_count:
                minority_count = count
                minority_class = i
        labels = (labels==minority_class).astype(int)
        

    majority = np.array([x for x in idx_train if labels[x] == 0])
    minority = np.array([x for x in idx_train if labels[x] == 1])

    num_minority = minority.shape[0]
    num_majority = majority.shape[0]
    print("Number of majority: ", num_majority)
    print("Number of minority: ", num_minority)

    minority_test = np.array([x for x in idx_test if labels[x] == 1])
    minority_all = np.hstack((minority, minority_test))

    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    majority = torch.LongTensor(majority)
    minority = torch.LongTensor(minority)
    minority_all = torch.LongTensor(minority_all)

    return features, labels, idx_train, idx_test, majority, minority, minority_all

def load_adj(dataset, n_nodes):
    path = "./dataset/" + dataset + "/"
    edges = np.genfromtxt("{}edges.{}".format(path, dataset), dtype=np.int32)

    adj_real = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(n_nodes, n_nodes),
                              dtype=np.float32)

    adj_norm = adj_real + adj_real.T.multiply(adj_real.T > adj_real) - adj_real.multiply(adj_real.T > adj_real)

    adj_norm = normalize(adj_norm + sp.eye(adj_norm.shape[0]))
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    return adj_real, adj_norm

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel() 
    return (tp*tn/((tp+fn)*(tn+fp)))**0.5
  
def accuracy(output, labels, output_AUC):
    preds = output.max(1)[1].type_as(labels)

    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    AUC = roc_auc_score(labels.cpu().numpy(), output_AUC.detach().cpu().numpy()) 
    conf = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    con_mat = list(conf.ravel())
    gmean = conf_gmean(conf)
    
    return f1, AUC, con_mat, gmean

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def add_edges(adj_real, adj_new):
    adj = adj_real+adj_new
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_norm = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    return adj, adj_norm


def generate_fake_nodes(n_real, n_fake):
    generate_nodes = torch.arange(n_real, n_real+n_fake)
    generate_label = torch.ones(n_fake, dtype=torch.long)
    return generate_nodes, generate_label

def to_cuda(*args, device):
    return tuple(x.to(device) for x in args)

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist