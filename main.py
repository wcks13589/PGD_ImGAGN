import argparse
from tkinter import Y
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from tqdm import trange

from utils import load_adj, load_data, accuracy, add_edges, to_cuda, generate_fake_nodes, euclidean_dist
from models import GCN, Generator
from new_pgd import NewPGDAttack

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs_G', type=int, default=100,
                    help='Number of epochs to train for gen.')
parser.add_argument('--epochs_D', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--ratio_generated', type=float, default=1,
                    help='ratio of generated nodes.')
parser.add_argument('--dataset', choices=['cora', 'pubmed', 'dblp', 'wiki', 'yelp'], default='cora')

parser.add_argument('--loss_type', type=str, default='tanhMarginMCE', choices=['CE', 'MCE', 'tanhMarginMCE', 'CL', 'MCL', 'tanhMarginMCL'])

args = parser.parse_args([])
# device = torch.device('cpu') 
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.dataset=='wiki':
    lr=0.001
elif args.dataset=='dblp':
    lr=0.0009
else:
    lr=0.01

if args.dataset == 'cora':
    weight_decay = 0.0001
elif args.dataset == 'citeseer':
    weight_decay = 0.0005
elif args.dataset == 'pubmed':
    weight_decay = 0.00008
elif args.dataset == 'dblp':
    weight_decay = 0.003
elif args.dataset == 'wiki':
    weight_decay = 0.0005
else:
    weight_decay = 0.0001


# Load Data
features, labels, \
    idx_train, idx_test, \
        majority, minority, minority_all = load_data(args.dataset)

n_real = labels.shape[0]
n_fakes = int(args.ratio_generated*len(majority)) - len(minority)
n_minority_all = minority_all.shape[0]

adj_real, adj_norm = load_adj(args.dataset, n_nodes=n_real+n_fakes)

# Generate fake nodes 
fake_nodes, fake_labels = generate_fake_nodes(n_real, n_fakes)

# Split train val data
idx_train, idx_val = train_test_split(idx_train, test_size=0.1)
idx_train = torch.cat([idx_train, fake_nodes])
labels = torch.cat([labels, fake_labels])

# labels for Generator
labels_true_G = torch.LongTensor(n_fakes).fill_(0) # Model_G產生的Fake Nodes不要被認出來
labels_min_G = torch.LongTensor(n_fakes).fill_(1) # Model_G產生的Fake Nodes要像minority

# labels for Discrminator
labels_true_D = torch.cat([torch.LongTensor(n_real-len(idx_test)-len(idx_val)).fill_(0), 
                           torch.LongTensor(n_fakes).fill_(1)])


# Initialize models and optimizers
model_G = Generator(n_minority_all)
model_D = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout,
              generate_node=fake_nodes,
              min_node = minority)

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr, weight_decay=weight_decay)

model_G, model_D, \
    features, adj_norm, \
        labels, labels_true_G, labels_min_G, labels_true_D, \
            idx_train, idx_test = \
        to_cuda(model_G, model_D,
        features, adj_norm,
        labels, labels_true_G, labels_min_G, labels_true_D,
        idx_train, idx_test, 
        device=device)


# Initialize Attacker
n_attackers = int(len(minority))
n_perturbations = n_attackers
attackers = np.random.choice(minority, size=n_attackers, replace=False)
attacker = NewPGDAttack(model=model_D, n_samples=n_real+n_fakes, n_real=n_real, n_attackers=n_attackers,
                        loss_type=args.loss_type, device=device).to(device)


max_f1 = 0
test_f1 = 0
test_AUC = 0
test_gmean = 0
test_con = []


'''
用到的參數
features 尚未加入fake nodes
minority (train)

'''

def log_attack(log_list, log_value, after_attack:bool):
    log_list.append(log_value)


def Evaluate(features, adj, after_attack):
    model_D.eval()
    output, output_gen, output_AUC = model_D(features, adj)

    ## attacker
    n_attack_success = n_attackers - torch.argmax(output[attackers], -1).sum().item()
    print(f'num of suceessful attack: {n_attack_success}')
    print(f'num of attackers: {n_attackers}')
    print(f'attack_rate: {n_attack_success/n_attackers:.2f}')

    history_n_attack_success.append(n_attack_success)
    history_attack_rate.append(n_attack_success/n_attackers)
    if after_attack:
        history_after_attack.append(1)
    else:
        history_after_attack.append(0)
        
    ## other_nodes
    test_f1, test_AUC, test_con, test_gmean = accuracy(output[idx_test], 
                                                        labels[idx_test], 
                                                        output_AUC[idx_test])
    print("Test f1: ", test_f1)
    print("Test AUC: ", test_AUC)
    print("Test confusion ", test_con)



adj_after_attack = adj_real # 已經加入fake node 但還沒加入fake edge
history_graph = []

## record history of attack
history_after_attack = []
history_n_attack_success = []
history_attack_rate = []


for epoch_G in range(args.epochs_G):
    ### Train Generator
    model_G.train()
    optimizer_G.zero_grad()
    z = Variable(torch.FloatTensor(np.random.normal(size=(n_fakes, 100)))).to(device)

    adj_min = model_G(z)
    fake_features_train, adj_temp_train = \
        model_G.generate_fake_features(adj_min, 
                                       features, 
                                       minority, 
                                       fake_nodes)
    fake_features_test, adj_temp_test = \
        model_G.generate_fake_features(adj_min, 
                                       features, 
                                       minority_all, 
                                       fake_nodes)

    adj_unnorm_train, adj_new_train = add_edges(adj_after_attack, adj_temp_train)
    adj_unnorm_test, adj_new_test = add_edges(adj_after_attack, adj_temp_test)

    adj_new_train = adj_new_train.to(device)
    adj_new_test = adj_new_test.to(device)

    features_new_train = torch.cat([features, fake_features_train.data])
    features_new_test = torch.cat([features, fake_features_test.data])

    output, output_gen, output_AUC = model_D(features_new_train, adj_norm)
    
    loss_g = F.nll_loss(output_gen[fake_nodes], labels_true_G) \
             + F.nll_loss(output[fake_nodes], labels_min_G) \
             + euclidean_dist(features[minority], fake_features_train).mean()
    loss_g.backward()
    optimizer_G.step()

    ### Train Discrminator
    for epoch_D in trange(args.epochs_D):
        model_D.train()
        optimizer_D.zero_grad()

        output, output_gen, output_AUC = model_D(features_new_train, adj_new_train) # train
        
        loss_dis = - euclidean_dist(features_new_train[minority], 
                                    features_new_train[majority]).mean()

        loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + \
                     F.nll_loss(output_gen[idx_train], labels_true_D) + \
                     loss_dis

        loss_train.backward()
        optimizer_D.step()

        # Evaluate
        # if not args.fastmode:
        #     model_D.eval()
        #     output, output_gen, output_AUC = model_D(features_new_train, adj_new_train)
            
        # f1_train, AUC_train, con_train, gmean_train = accuracy(output[idx_train], 
        #                                                        labels[idx_train], 
        #                                                        output_AUC[idx_train])

        # f1_val, AUC_val, con_val, gmean_val = accuracy(output[idx_val], 
        #                                                labels[idx_val], 
        #                                                output_AUC[idx_val])

        # if max_f1 < f1_val:
        #     output, output_gen, output_AUC = model_D(features_new_test, adj_new_test)
        #     test_f1, test_AUC, test_con, test_gmean = accuracy(output[idx_test], 
        #                                                        labels[idx_test], 
        #                                                        output_AUC[idx_test])
        #     max_f1 = f1_val
        #     print("Test f1: ", test_f1)
        #     print("Test AUC: ", test_AUC)
        #     print("Test gmean: ", test_gmean)
        #     print("Test confusion ", test_con)

    # print("Epoch:", '%04d' % (epoch_G + 1), 
    #     "train_f1=", "{:.5f}".format(f1_train), "train_AUC=", "{:.5f}".format(AUC_train),"train_gmean=", "{:.5f}".format(gmean_train),
    #     "val_f1=", "{:.5f}".format(f1_val), "val_AUC=", "{:.5f}".format(AUC_val),"val_gmean=", "{:.5f}".format(gmean_val))
    
    history_graph.append(adj_new_train) # added fake nodes and fake edges and do normalization
    print(f'epoch:{epoch_G+1}  =======================Round 1 Evaluate=======================')
    Evaluate(features_new_train, history_graph[-1], after_attack=False)


    adj_unnorm_train, adj_new_train = add_edges(adj_real, adj_temp_train)
    loss_log = attacker.attack(features_new_train,
                                adj_unnorm_train.to_dense(), 
                                labels, attackers, n_perturbations)

    print(f'epoch:{epoch_G+1}  =======================Round 2 Evaluate=======================')
    Evaluate(features_new_train, attacker.modified_adj_norm, after_attack=True)
    edge_index = attacker.modified_adj.nonzero().cpu().T 
    mask = edge_index < n_real
    mask = torch.logical_and(mask[0], mask[1])
    edge_index = edge_index[:, mask]

    adj_after_attack = sp.coo_matrix((np.ones(mask.sum()), (edge_index[0], edge_index[1])),
                                     shape=(attacker.nnodes, attacker.nnodes),
                                     dtype=np.float32)



