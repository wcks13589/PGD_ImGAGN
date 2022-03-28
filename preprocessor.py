import numpy as np
from scipy.io import loadmat

class preprocessor:
    def __init__(self, dataset, data_mat, edge_type):
        self.dataset = dataset
        self.edge_type = edge_type
        self.path = f'./dataset/{self.dataset}/'
        self.data = loadmat(data_mat)
        
        if data_mat == './dataset/yelp/YelpChi.mat':
            self.edge_types = ['net_rur', 'net_rtr', 'net_rsr']
        
        self.edges = self.load_edge()
        self.feat = self.load_feat()
        self.label = self.load_label()
        
    def load_edge(self):
        
        edge_dict = {}
        for edge_type in self.edge_types:
            edge_dict[edge_type] = self.data[edge_type].nonzero()
        return edge_dict
    
    def load_feat(self):
        return self.data['features'].todense().A
    
    def load_label(self):
        return self.data['label'].flatten()
    
    def convert_edge(self):
        edge = self.edges[self.edge_types[self.edge_type]]
        with open(self.path + f'edges.{self.dataset}', 'w') as f:
            for i, j in zip(edge[0], edge[1]):
                f.writelines(f'{i} {j}\n')
                
    def convert_feat(self):
        output = np.concatenate([self.feat,self.label.reshape(-1,1)], 1)
        np.savetxt(self.path + f'features.{self.dataset}', 
                   output, delimiter=' ')
    
    def split_train_test_nodes(self):
        nodes = np.arange(self.feat.shape[0])
        np.random.shuffle(nodes)
        train_nodes = nodes[:-18474]
        test_nodes = nodes[-18474:]
        np.savetxt(self.path + f'train.{self.dataset}', train_nodes, fmt='% 4d')
        np.savetxt(self.path + f'test.{self.dataset}', test_nodes, fmt='% 4d')

if __name__ == '__main__':
    data = preprocessor('yelp', './dataset/yelp/YelpChi.mat', 2)
    data.convert_edge()
    data.convert_feat()
    data.split_train_test_nodes()