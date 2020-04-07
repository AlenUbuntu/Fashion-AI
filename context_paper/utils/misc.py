import numpy as np 
import os
import torch
import scipy.sparse as sp

def normalize_feature(feats, mean=None, std=None, return_moments=False):
    """
    Expect input feats matrix to be a sparse feature matrix
    """
    if mean is None:
        mean = feats.mean(axis=0)
    else:
        mean = mean
    if std is None:
        square = feats.power(n=2)
        e_square = square.mean(axis=0)
        std = np.power(e_square - np.power(mean,2), 0.5)
    else:
        std = std
    # TODO: the following step is memory expensive since it converts a big sparse 
    # matrix into a dense matrix in the computation. A better way to do it is just
    # compute the mean and variance, and normalize each row when necessary.
    res = (feats - mean) / std 
    if return_moments:
        return res, mean, std 
    return res

def compute_degree_support(adj, S, adj_self_connections=False, verbose=True):
    if verbose:
        print("Compute adjacency matrix up to {} degrees, i.e., A^{}".format(S, S))
    n_nodes = adj.shape[0]
    # [A0, A1, A2, ..., AS]
    supports = [sp.identity(n_nodes)]
    
    if S == 0:
        # only consider A0
        return supports
    
    # include A0 and A1, A0 = I, A1 = A + I (if include self connection else 0)
    supports = [sp.identity(n_nodes), adj.astype(np.float64) + adj_self_connections * sp.identity(n_nodes)]

    prev = adj 
    for _ in range(S-1):
        pow = prev.dot(adj)  # A^n = A^(n-1) * A
        # (A^n)_{i,j} reflects the number of n-hop paths connecting node i and j
        # if self-connection is allowed, we can move <=n-1 steps on a node itself and then move to the target node
        # Create a subgraph where nodes are connected by only 1 n-hop path.
        new_adj = (pow == 1).astype(np.float64)
        new_adj.setdiag(0) # remove self-connection
        new_adj.eliminate_zeros()
        supports.append(new_adj)
        prev = pow
        
    return supports

def normalize_nonsym_adj(adj):
    """
    Normalize adjacency matrix, so that summation of edge weights of neighbors is either 1 or 0, i.e., 
    the out degree = 1 or 0
    """
    degree = np.asarray(adj.sum(1)).flatten()

    # set zeros to inf to avoid divided by 0
    degree[degree==0.] = np.inf 

    # compute inverse of degrees 
    degree_inverse_sqrt = 1./np.sqrt(degree)  # 1./sqrt(D)
    degree_inverse_mat = sp.diags([degree_inverse_sqrt], [0])
    degree_inverse_mat = degree_inverse_mat.dot(degree_inverse_mat)  # 1./D

    # normalize
    adj_norm = degree_inverse_mat.dot(adj)

    return adj_norm

def support_dropout(adj, p, drop_edge=False):
    assert 0.0 < p < 1.0
    lower_adj = sp.tril(adj)
    n_nodes = lower_adj.shape[0]

    # find nodes to isolate 
    isolate = np.random.choice(range(n_nodes), size=int(n_nodes * p), replace=False)
    s_idx, e_idx = lower_adj.nonzero()

    # mask the nodes that have been selected
    # here mask contains all the nodes that have been selected in isolated
    # regardless whether it is source node or end node of an edge
    mask = np.in1d(s_idx, isolate)
    mask += np.in1d(e_idx, isolate)
    # csr_matrix.data is the array storing the non-zero elements, it is usually much 
    # fewer than csr_matrix.shape[0] * csr_matrix.shape[1]
    assert mask.shape[0] == lower_adj.data.shape[0]

    lower_adj.data[mask] = 0 
    lower_adj.eliminate_zeros()

    if drop_edge:
        prob = np.random.uniform(0, 1, size=lower_adj.data.shape)
        remove = prob < p 
        lower_adj.data[remove] = 0 
        lower_adj.eliminate_zeros()
    
    lower_adj = lower_adj + lower_adj.transpose()
    return lower_adj

def csr_to_sparse_tensor(matrix):
    coo = matrix.tocoo()
    shape = matrix.shape 
    index = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
    value = torch.FloatTensor(coo.data.astype(np.float64))
    tensor = torch.sparse.FloatTensor(index, value, torch.Size(shape))
    return tensor

if __name__ == '__main__':
    from scipy.sparse import load_npz
    path = '/home/alan/Downloads/fashion/polyvore/dataset/X_test.npz'
    x = load_npz(path)
    normalize_feature(x)
    
    a = csr_to_sparse_tensor(x)
    print(a)