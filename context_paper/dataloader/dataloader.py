import numpy as np 
import abc
from utils.misc import *

class DataLoader(metaclass=abc.ABCMeta):
    def __init__(self, path):
        """
        Abstract dataloader - support common utility functions
        """
        self.data_path = path 

        print('Initializing dataloader ... ')

        self.init_phase('train')
        self.init_phase('valid')
        self.init_phase('test')
        print()
    
    @abc.abstractclassmethod
    def init_phase(self, phase):
        """
        init_phase method must be implemented by subclasses.
        """
        pass 
    
    def normalize_feature(self, feats, mean=None, std=None, return_moments=False):
        return normalize_feature(feats, mean, std, return_moments)
    
    def get_negative_training_edges(self, num_nodes, lower_adj):
        """
        This implementation samples a negative edge from a graph. However, it may sample
        duplicate edges from time to time. It is a severe issue for small graphs, but could be ok
        for large graphs, since the duplicate probability is pretty low.
        """
        u = np.random.randint(num_nodes)
        v = np.random.randint(num_nodes)

        # check if (u, v) or (v, u) is an positive edge. If so, find another edge that is negative.
        # note that (u, v) and (v, u) represent the same undirectional edge, and 
        # lower_adj is only the lower triangular part of the full symmetric adjacency matrix 
        # only one of adj[u,v] or adj[v,u] may appear in lower_adj
        # Also lower_adj[u,u] = 1 for self-loop
        while lower_adj[u, v] == 1 or lower_adj[v, u] == 1:
            u = np.random.randint(num_nodes)
        
        return u, v