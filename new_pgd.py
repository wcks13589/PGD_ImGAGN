import numpy as np
from tqdm import trange
import scipy.sparse as sp

from deeprobust.graph import utils

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from pgd import PGDAttack

class NewPGDAttack(PGDAttack):
    
    def __init__(self, model, n_samples, n_real, n_attackers, loss_type='CE', attack_structure=True, attack_features=False, device='cpu'):

        super(PGDAttack, self).__init__(model, n_samples, device=device)

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        
        self.n_samples = n_samples
        self.n_real = n_real
        self.n_attackers = n_attackers
        
        # self.attack_structure = attack_structure
        # self.attack_features = attack_features
        
        # if attack_structure:
        #     self.adj_changes = Parameter(torch.FloatTensor(int(self.nnodes*(self.nnodes-1)/2)))
        #     self.adj_changes.data.fill_(0)
        #     self.complementary_adj = None

        # if attack_features:
        #     self.feat_changes = Parameter(torch.FloatTensor(n_samples, n_features))
        #     self.feat_changes.data.fill_(0)
        #     self.complementary_feat = None

        if n_attackers != 0:
            self.adj_changes = Parameter(torch.FloatTensor(n_attackers, n_real-n_attackers))
            self.adj_changes.data.fill_(1)
            self.complementary_adj = None



    def attack(self, ori_features, ori_adj, labels, attackers, n_perturbations, epochs=200, **kwargs):
        '''
        ori_features (include fake nodes) (normalized)
        ori_adj (include fake nodes)

        '''
        victim_model = self.surrogate
        self.attackers = attackers
        self.benign_nodes = list(set(range(self.n_real)) - set(self.attackers))
        
        self.sparse_features = sp.issparse(ori_features)
        ori_adj = ori_adj.to(self.device)

        if utils.is_sparse_tensor(ori_adj):
            ori_adj_norm = utils.normalize_adj_tensor(ori_adj, sparse=True)
        else:
            ori_adj_norm = utils.normalize_adj_tensor(ori_adj)

        victim_model.eval()
        for t in trange(epochs):
            # if self.attack_structure:
            #     modified_adj = self.get_modified_adj(ori_adj)
            #     adj_norm = utils.normalize_adj_tensor(modified_adj)
            # else:
            #     adj_norm = ori_adj_norm
            
            # if self.attack_features:
            #     modified_feat = self.get_modified_feat(adj_feat)
            #     features_norm = utils.normalize_adj_tensor(modified_feat)[:self.n_samples, self.n_samples:]
            # else:
            #     features_norm = ori_features

            modified_adj = self.get_attacked_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            
            features_norm = ori_features

            loss, _ = self.my_loss(victim_model, ori_features, features_norm, ori_adj_norm, adj_norm, labels)
            
            if 'CE' in self.loss_type:
                lr = 200 / np.sqrt(t+1)
            elif 'CL' in self.loss_type:
                lr = 2 / np.sqrt(t+1)

            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)
            

        loss_log = self.random_sample(ori_adj, ori_adj_norm, ori_features, labels, n_perturbations)

        self.modified_adj = self.get_attacked_adj(ori_adj).detach()
        self.modified_adj_norm = utils.normalize_adj_tensor(self.modified_adj)
        self.check_adj_tensor(self.modified_adj)
        
        return loss_log
    
    def projection(self, n_perturbations):
        if self.n_attackers != 0:
            if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
                left = (self.adj_changes - 1).min()
                right = self.adj_changes.max()
                miu = self.bisection(left, right, n_perturbations, self.adj_changes, epsilon=1e-5)
                self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
            else:
                self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_attacked_adj(self, ori_adj):

        if self.complementary_adj is None:
            self.complementary_adj = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        benign_nodes = list(set(range(self.n_real)) - set(self.attackers))
        m[self.attackers.repeat(len(benign_nodes)), benign_nodes*self.n_attackers] = self.adj_changes.flatten()
        m = m + m.t()
        
        modified_adj = self.complementary_adj * m + ori_adj

        return modified_adj

    def get_modified_adj(self, ori_adj):

        if self.complementary_adj is None:
            self.complementary_adj = (torch.ones(self.nnodes,self.nnodes) - torch.eye(self.nnodes) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tuple(tril_indices)] = self.adj_changes
        m = m + m.t()

        # m *= self.node_importance

        # m = (m+m.T)/2

        modified_adj = self.complementary_adj * m + ori_adj

        return modified_adj
    
    def get_modified_feat(self, ori_adj):

        nnodes = self.n_samples+self.n_features
        
        if self.complementary_feat is None:
            self.complementary_feat = (torch.ones_like(ori_adj) - torch.eye(nnodes).to(self.device) - ori_adj) - ori_adj
        
        m = torch.zeros((nnodes, nnodes)).to(self.device)
        m[:self.n_samples, self.n_samples:] = self.feat_changes
        m = m + m.t()
        modified_adj = self.complementary_feat * m + ori_adj

        return modified_adj
    
    def random_sample(self, ori_adj, ori_adj_norm, ori_features, labels, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            if self.n_attackers != 0:
                s_adj = self.adj_changes.cpu().detach().numpy()
                
            for i in range(K):
                sample_sum = 0
                sampled_adj = np.random.binomial(1, s_adj)
                sample_sum += sampled_adj.sum()
            
                if sample_sum > n_perturbations:
                    continue
                
                if self.n_attackers != 0:
                    self.adj_changes.data.copy_(torch.tensor(sampled_adj))
                    modified_adj = self.get_attacked_adj(ori_adj)
                    adj_norm = utils.normalize_adj_tensor(modified_adj)
                else:
                    adj_norm = ori_adj_norm
                    
                features_norm = ori_features
                    
                # 有改過
                loss, loss_log = self.my_loss(victim_model, ori_features, features_norm, ori_adj_norm, adj_norm, labels)
                
                
                if best_loss < loss:
                    best_loss = loss
                    best_s_adj = sampled_adj
            
            self.adj_changes.data.copy_(torch.tensor(best_s_adj))

        return loss_log
    
    def bisection(self, a, b, n_perturbations, changes, epsilon):
        def func(x):
            return torch.clamp(changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    def my_loss(self, model, ori_features, mod_features, ori_adj, mod_adj, labels):
        
        output_ori = model.get_embedding(ori_features, ori_adj)
        output_per = model.get_embedding(mod_features, mod_adj)

        logits = model(mod_features, mod_adj)[0][self.attackers]
        labels = labels[self.attackers]
        
        loss_log = {}
        
        if 'MC' in self.loss_type:
            not_flipped = logits.argmax(-1) == labels
        else:
            not_flipped = torch.ones_like(labels).bool()
            
        loss_CE = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        
        criterion = InfoNCE(0.2, False)
        loss_CL = criterion(output_ori[self.attackers], output_per[self.attackers])[not_flipped].mean()
        
        loss_log['CE'], loss_log['CL'] = loss_CE.item(), loss_CL.item()
        
        if 'CE' in self.loss_type:
            loss = loss_CE
            
        elif 'CL' in self.loss_type:
            loss = loss_CL
            
        if 'tanhMargin' in self.loss_type:
            alpha = 1

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            
            margin = torch.tanh(-margin)
            
            loss = alpha * margin.mean() + (1 - alpha) * loss
        
        return loss, loss_log
    
def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()
    
class InfoNCE():
    def __init__(self, tau, intraview_negs=True):
        self.tau = tau
        self.intraview_negs = intraview_negs
        
    def add_intraview_negs(self, anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)                     # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)     # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)     # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask
    
    def sample(self, anchor, sample):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        
        if self.intraview_negs:
            anchor, sample, pos_mask, neg_mask = self.add_intraview_negs(anchor, sample, pos_mask, neg_mask)

        return anchor, sample, pos_mask, neg_mask
    
    def add_extra_mask(self, pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
        if extra_pos_mask is not None:
            pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
        if extra_neg_mask is not None:
            neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
        else:
            neg_mask = 1. - pos_mask
        return pos_mask, neg_mask

    def __call__(self, anchor, sample, extra_pos_mask=None, extra_neg_mask=None):
        
        anchor1, sample1, pos_mask1, neg_mask1 = self.sample(anchor=anchor, sample=sample)
        anchor2, sample2, pos_mask2, neg_mask2 = self.sample(anchor=sample, sample=anchor)
        
        pos_mask1, neg_mask1 = self.add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = self.add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2)

        return (l1 + l2) * 0.5
    
    def loss(self, anchor, sample, pos_mask, neg_mask):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        
        # extra_mask = torch.zeros_like(exp_sim, device=exp_sim.device)
        # extra_mask[torch.arange(exp_sim.size(0)), torch.argmin(exp_sim, 1)] = 1
    
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

        return -loss