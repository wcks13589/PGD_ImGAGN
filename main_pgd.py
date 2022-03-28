import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from new_pgd import NewPGDAttack
import argparse

from data import GraphData

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=5168, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--loss_type', type=str, default='CL', choices=['CE', 'MCE', 'tanhMarginMCE', 'CL', 'MCL', 'tanhMarginMCL'])
parser.add_argument('--attack_graph', action="store_true", default=True)
parser.add_argument('--attack_feat', action="store_false", default=False)
args = parser.parse_args([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = GraphData(args.dataset, normalize=False)

adj, features, labels,idx_train, idx_val, idx_test = data.adj, data.x, data.y, data.idx_train, data.idx_val, data.idx_test

perturbations = int(args.ptb_rate * torch.div(adj.sum(), 2))

x_feat, edge_index_feat, edge_attr_feat = data.load_feature_graph()
adj_feat = torch.sparse_coo_tensor(edge_index_feat, edge_attr_feat, [np.sum(features.shape)]*2).to_dense()

n_samples, n_features = features.shape

features_norm = torch.FloatTensor(normalize_feature(features))

def test(new_adj, new_features, gcn=None):
    ''' test on GCN '''

    if gcn is None:
        # adj = normalize_adj_tensor(adj)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(new_features, new_adj, labels, idx_train, idx_val, patience=30) # train with validation model picking
        gcn.eval()
        output = gcn.predict().cpu()
    else:
        gcn.eval()
        output = gcn.predict(new_features.to(device), new_adj.to(device)).cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    result = {'y_true':labels[idx_test].detach().numpy(), 'y_pred':output[idx_test].detach().max(1)[1].type_as(labels).numpy(),
              'acc':acc_test.item()}
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()), args.loss_type)

    return result


def main(args):
    target_gcn = GCN(nfeat=features.shape[1],
                     nhid=16,nclass=labels.max().item() + 1,
                     dropout=0.5, device=device)

    target_gcn = target_gcn.to(device)
    target_gcn.fit(features_norm, adj, labels, idx_train)#, idx_val)#, patience=30)
    # target_gcn.fit(features, adj, labels, idx_train)

    print('=== testing GCN on clean graph ===')
    args.loss_type = ''
    clean_result = test(adj, features_norm, target_gcn)
    clean_results.append(clean_result['acc'])

    for loss_type in args.loss_types:
        args.loss_type = loss_type
        print('=== args setting ===')
        print(f'ptb_rate:{args.ptb_rate}, loss_type:{args.loss_type}')

        # Setup Attack Model
        print('=== setup attack model ===')
        model = NewPGDAttack(model=target_gcn, n_samples=n_samples, n_features=n_features,
                             attack_structure=args.attack_graph, attack_features=args.attack_feat,
                             loss_type=args.loss_type, device=device)
        model = model.to(device)
    
        # model.attack(features, adj, labels, idx_train, perturbations, epochs=args.epochs)
        # Here for the labels we need to replace it with predicted ones
        fake_labels = target_gcn.predict(features_norm.to(device), adj.to(device))
        fake_labels = torch.argmax(fake_labels, 1).cpu()
        # Besides, we need to add the idx into the whole process
        idx_fake = np.concatenate([idx_train,idx_test])
        
        loss_log = model.attack(features_norm, adj, adj_feat, fake_labels, idx_fake, perturbations, epochs=args.epochs)
    
        print('=== testing GCN on Evasion attack ===')
        
        if args.attack_graph:
            modified_adj = model.modified_adj
        else:
            modified_adj = adj
        if args.attack_feat:
            modified_feat = model.modified_feat[:n_samples, n_samples:]
        else:
            modified_feat = features_norm
        print("========================================Train set results:",
              "loss_CE= {:.4f}".format(loss_log['CE']),
              "loss_CL= {:.4f}".format(loss_log['CL']))
        evasion_result = test(modified_adj, modified_feat, target_gcn)
        
    
        # modified_features = model.modified_features
        print('=== testing GCN on Poisoning attack ===')
        poison_result = test(modified_adj, modified_feat)

        # # if you want to save the modified adj/features, uncomment the code below
        # model.save_adj(root='./', name=f'mod_adj')
        # model.save_features(root='./', name='mod_features')

        evasion_results[loss_type].append(evasion_result['acc'])
        poison_results[loss_type].append(poison_result['acc'])
    
    return adj, modified_adj, target_gcn, clean_result, evasion_result, poison_result



def print_average_result(results:dict, Evaluate_task:str):
    print(f'================={Evaluate_task}=================')
    for loss_type, acc in results.items():
        print(f'When loss type = {loss_type}, Average Accuracy: {np.mean(acc)}')



if __name__ == '__main__':

    seeds = [15, 1234, 1012, 888, 426]
    args.loss_types = ['CE', 'tanhMarginMCE', 'CL']
    
    clean_results = []
    evasion_results = {loss_type:[] for loss_type in args.loss_types}
    poison_results = {loss_type:[] for loss_type in args.loss_types}

    for seed in seeds:
        # args.ptb_rate = rate
        # perturbations = int(args.ptb_rate * (adj.sum()//2))
        # print(f'now the ptb_rate is {rate}')
        print(f'now the seed is {seed}')
        args.seed = seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device != 'cpu':
            torch.cuda.manual_seed(args.seed)
        
        adj, modified_adj, gcn, clean_result, evasion_result, poison_result = main(args)

    print(f'===================Clean_results===================\n{np.mean(clean_results)}')
    print_average_result(evasion_results, 'Evasion')
    print_average_result(poison_results, 'Poison')
        








