import argparse
import torch
from utils.misc import normalize_nonsym_adj, compute_degree_support, csr_to_sparse_tensor
from model.gae import CompatibilityGAE
from dataloader import DataLoaderPolyvore



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Fill in the Blank Questions")
    parser.add_argument('-eo', '--expand-outfit', action='store_true',
                        help='whether to expand edges for question item and answer nodes or not')
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
    EO = args['expand_outfit']
    RESAMPLE = args['resample']
    SUBSET = args['subset']
    DEVICE = args['device']
    K = args['K']

    # these parameters should be set same as those in training phase
    # load weight
    checkpoint = torch.load(args['load_model'])
    # model_args = checkpoint['args']

    model_args = args
    checkpoint = {'state_dict': checkpoint}

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

    # get support 
    test_support = compute_degree_support(test_mp_adj, DEGREE, ADJ_SELF_CONNECTIONS)

    # normalize these support adjacency matrices, except the first one which is symmetric
    for i in range(1, len(test_support)):
        test_support[i] = normalize_nonsym_adj(test_support[i])
    
    # convert features to tensors
    test_features = torch.from_numpy(test_features).to(DEVICE)
    test_labels = torch.from_numpy(test_labels).float().to(DEVICE)

    test_row_idx = torch.from_numpy(test_row_idx).long().to(DEVICE)
    test_col_idx = torch.from_numpy(test_col_idx).long().to(DEVICE)

    test_support = [csr_to_sparse_tensor(adj).to(DEVICE) for adj in test_support]

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

    # evaluate or predict
    with torch.no_grad():
        test_pred = model.predict(test_features, test_support, test_row_idx, test_col_idx)
        test_acc = model.accuracy(test_pred, test_labels)

        print("Test Acc on link prediction: {:.4f}".format(test_acc))

        # create test question iterator
        iterator = dl.yield_test_questions_K_edges(resample=RESAMPLE, K=K, subset=SUBSET, expand_outfit=EO)

        correct = 0.
        total = 0.
        for i, (question_mp_adj, ques_idx, ans_idx, labels) in enumerate(iterator):
            print("Processed Samples: {}".format(i+1), end='\r')
            # get query support
            query_support = compute_degree_support(question_mp_adj, DEGREE, ADJ_SELF_CONNECTIONS, verbose=False)
            # normalize non-symmetric adj in query support
            for i in range(1, len(query_support)):
                query_support[i] = normalize_nonsym_adj(query_support[i])
            
            # convert to tensors
            query_support = [csr_to_sparse_tensor(adj).to(DEVICE) for adj in query_support]
            ques_idx = torch.from_numpy(ques_idx).long().to(DEVICE)
            ans_idx = torch.from_numpy(ans_idx).long().to(DEVICE)

            pred_prob = model.predict_prob(test_features, query_support, ques_idx, ans_idx)
            pred_prob = pred_prob.cpu().numpy()  # predicted link probability for edges between each question item and each answer item
            pred_prob = pred_prob.reshape(-1, 4)  # each answer set contains 4 choices (N, 4) where N is the number of question items
            pred_avg_prob = pred_prob.mean(axis=0)  # average link probability of each answer item 
            selected_item = pred_avg_prob.argmax()  # select the item with highest average link probability

            # note that in labels, label == 1 if an edge connects an question item and an answer item
            ground_truth_prob = labels.reshape(-1, 4).mean(axis=0)  # average ground truth prob  of each answer item. It will be 1 for correct
            # answer, and 0 otherwise.
            gt = ground_truth_prob.argmax()
            correct += int(selected_item == gt)
            total += 1
        
        print("Overall FITB Acc: {:.4f}".format(correct/total))
