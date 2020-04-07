import argparse 
import os 
import json
import torch
import scipy.sparse as sp
import numpy as np 
from dataloader import DataLoaderPolyvore
from utils.misc import compute_degree_support, normalize_nonsym_adj, csr_to_sparse_tensor
from model import CompatibilityGAE
from sklearn.metrics import roc_auc_score
import faulthandler
faulthandler.enable()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Compatility of an Outfit.")
    parser.add_argument('-resample', '--resample', action='store_true',
                        help='Runs the test with the resampled FITB tasks (harder)')
    parser.add_argument('-subset', '--subset', action='store_true',
                        help='if True, use only a subset of outfits as the query, and use the rest as links to the choices.')
    parser.add_argument('-load', '--load-model', type=str,
                        default='/home/alan/Downloads/fashion/polyvore/best_model.pth')
    parser.add_argument('-d', '--dataset', type=str, default='polyvore', choices=['polyvore', 'fashiongen'])
    parser.add_argument('--data-dir', type=str,
                        default='/home/alan/Downloads/fashion/polyvore/dataset2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-deg', '--degree', type=int, default=1, help='degree of convolution, i.e., number of support nodes.')
    parser.add_argument('-hi', '--hidden', type=int, nargs='+', default=[350, 350, 350], help='Number of hidden units in the GCN layers.')
    parser.add_argument('-drop', '--dropout', type=float, default=0.5, help='dropout probability')
    mg = parser.add_mutually_exclusive_group(required=False)
    mg.add_argument('-bn', '--batch-norm', action='store_true',
                    help='Option to turn on batchnorm in GCN layers')
    mg.add_argument('-no-bn', '--no-batch-norm', action='store_false',
                    help='Option to turn off batchnorm in GCN layers')
    parser.set_defaults(batch_norm=True)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0., help='weight decay')
    parser.add_argument('-K', '--K', type=int, default=1,
                        help='number of edges to expand for each question node')

    args = parser.parse_args()
    args = vars(args)
    print("\nSetting: \n")
    for key, val in args.items():
        print("{}: {}".format(key, val))
    print()

    # Define parameters
    DATASET = args['dataset']
    DATA_DIR = args['data_dir']
    RESAMPLE = args['resample']
    SUBSET = args['subset']
    DEVICE = args['device']
    K = args['K']

    # these parameters should be set same as those in training phase
    # load weight
    checkpoint = torch.load(args['load_model'])
    model_args = checkpoint['args']

    DEGREE = model_args['degree']  
    HIDDEN = model_args['hidden']
    DROP = model_args['dropout']
    BATCH_NORM = model_args['batch_norm']
    LR = model_args['learning_rate']
    WD = model_args['weight_decay']
    NUM_CLASSES = 2
    ADJ_SELF_CONNECTIONS = True

    # prepare dataset
    if DATASET in ('polyvore', 'fashiongen'):
        if DATASET == 'polyvore':
            dl = DataLoaderPolyvore(DATA_DIR)
        else:
            raise NotImplementedError('Support to fashiongen dataset will be added soon!')

        # node features, message-passing adj, ground-truth labels, start node idx, end node idx of edges to evaluate loss
        train_features, train_mp_adj, train_labels, train_row_idx, train_col_idx = dl.get_phase('train')
        test_features, test_mp_adj, test_labels, test_row_idx, test_col_idx = dl.get_phase('test')

        # normalize features
        train_features, mean, std = dl.normalize_feature(train_features, return_moments=True)
        test_features = dl.normalize_feature(test_features, mean=mean, std=std, return_moments=False)

    else:
        raise NotImplementedError('Dataloader for dataset {} is not supported yet!'.format(DATASET))
    
    # convert to tensor 
    test_features = torch.from_numpy(test_features).to(DEVICE)

    # get support 
    test_support = compute_degree_support(test_mp_adj, DEGREE, ADJ_SELF_CONNECTIONS)
    num_supports = len(test_support)
    settings = {
        'num_support': num_supports,  
        'dropout': DROP,
        'batch_norm': BATCH_NORM,
        'learning_rate': LR,
        'wd': WD,
    }
    # create model
    model = CompatibilityGAE(
        input_dim=test_features.shape[1],
        hidden=HIDDEN,
        num_classes=2,
        settings=settings
    )
    model.to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # read compatible file
    if not RESAMPLE:
        with open(os.path.join(DATA_DIR, 'compatibility_test.json')) as f:
            compat_data = json.load(f)
    else:
        with open(os.path.join(DATA_DIR, 'compatibility_RESAMPLED_test.json')) as f:
            compat_data = json.load(f)
    
    predictions = []
    ground_truth = []

    with torch.no_grad():
        # start prediction, the goal is to predict the link probability between outfit items
        for idx, outfit in enumerate(compat_data):

            # outfit = [[item1_idx, item2_idx, ...], label] where label = 1 if compatible else 0 
            items, label = outfit 
        
            num_unique_items = test_features.shape[0]
            # create message passing graph for this outfit
            adj = sp.csr_matrix((num_unique_items, num_unique_items))  # no connections, items are isolated from each other
            adj = adj.tolil()
         
            if SUBSET:  # use only a subset of (size 3) an outfit
                items = np.random.choice(items, size=3, replace=False)

            # find available edges in the graph 
            if K > 0:
                available_adj = dl.test_adj.copy().tolil()

                # remove edges connecting outfit items in the test graph
                for i, s_idx in enumerate(items[:-1]):
                    for e_idx in items[i+1:]:
                        available_adj[s_idx, e_idx] = 0
                        available_adj[e_idx, s_idx] = 0 
            
                available_adj = available_adj.tocsr()
                available_adj.eliminate_zeros()
        
                node2expand = np.unique(items)
                for node in node2expand:
                    # find K expand edges
                    edges =  dl.sample_K_edges(available_adj, node, K)

                    # add to message passing graph 
                    for u, v in edges:
                        adj[u, v] = 1
                        adj[v, u] = 1
        
            adj = adj.tocsr()

            # find edges to predict
            query_row_indices = []
            query_col_indices = []

            for i, s_idx in enumerate(items[:-1]):
                for e_idx in items[i+1:]:
                    query_row_indices.append(s_idx)
                    query_col_indices.append(e_idx)
        
            query_row_indices = np.array(query_row_indices)
            query_col_indices = np.array(query_col_indices)

            query_support = compute_degree_support(adj, DEGREE, ADJ_SELF_CONNECTIONS, verbose=False)
            for i in range(1, len(query_support)):
                query_support[i] = normalize_nonsym_adj(query_support[i])
            query_support = [csr_to_sparse_tensor(each).to(DEVICE) for each in query_support]

            query_row_indices = torch.from_numpy(query_row_indices).long().to(DEVICE)
            query_col_indices = torch.from_numpy(query_col_indices).long().to(DEVICE)

            pred_prob = model.predict_prob(test_features, query_support, query_row_indices, query_col_indices)
            pred_score = pred_prob.mean()

            print("Mean score of outfit {}: {:.4f} -- Expected: {:.4f}".format(idx, pred_score, label), end='\r')
            predictions.append(pred_score)
            ground_truth.append(label)
    print() 
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    auc = roc_auc_score(ground_truth, predictions)
    print("Overall AUC: {:.4f}".format(auc))
