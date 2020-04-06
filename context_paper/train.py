import argparse
import time 
import numpy as np 
import torch 
from pprint import pprint 
from dataloader import DataLoaderPolyvore
from model import CompatibilityGAE
from utils.misc import compute_degree_support, normalize_nonsym_adj, support_dropout, csr_to_sparse_tensor
from copy import deepcopy

# set random seed
# seed = int(time.time())  # 12342
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='visual compatibility')
parser.add_argument('-d', '--dataset', type=str, default='polyvore', choices=['polyvore', 'fashiongen'])
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='learning rate')
parser.add_argument('-wd', '--weight-decay', type=float, default=0., help='weight decay')
parser.add_argument('-e', '--epochs', type=int, default=4000, help='Number of training epochs')
parser.add_argument('-hi', '--hidden', type=int, nargs='+', default=[350, 350, 350], help='Number of hidden units in the GCN layers.')
parser.add_argument('-drop', '--dropout', type=float, default=0.5, help='dropout probability')
parser.add_argument('-deg', '--degree', type=int, default=1, help='degree of convolution, i.e., number of support nodes.')
parser.add_argument('-sd', '--support-dropout', type=float, default=0.15, help='Use dropout on support adjacency matrix, dropping all the connections from some nodes')
parser.add_argument('--data-dir', type=str,
                    default='/home/alan/Downloads/fashion/polyvore/dataset')
parser.add_argument('--device', type=str, default='cuda:0')

mg = parser.add_mutually_exclusive_group(required=False)
mg.add_argument('-bn', '--batch-norm', action='store_true',
                help='Option to turn on batchnorm in GCN layers')
mg.add_argument('-no-bn', '--no-batch-norm', action='store_false',
                help='Option to turn off batchnorm in GCN layers')
parser.set_defaults(batch_norm=True)

args = parser.parse_args()
args = vars(args)

print("\nSetting: \n")
for key, val in args.items():
    print("{}: {}".format(key, val))
print()

# Define parameters
DATASET = args['dataset']
NB_EPOCHS = args['epochs']
DROP = args['dropout']
HIDDEN = args['hidden']
LR = args['learning_rate']
NUM_CLASSES = 2  # each edge is either connected or not 
DEGREE = args['degree']
BATCH_NORM = args['batch_norm']
SUP_DROP = args['support_dropout'] 
ADJ_SELF_CONNECTIONS = True 
DATA_DIR = args['data_dir']
WD = args['weight_decay']
DEVICE = args['device']

# prepare dataset
if DATASET in ('polyvore', 'fashiongen'):
    if DATASET == 'polyvore':
        dl = DataLoaderPolyvore(DATA_DIR)
    else:
        raise NotImplementedError('Support to fashiongen dataset will be added soon!')

    # node features, message-passing adj, ground-truth labels, start node idx, end node idx of edges to evaluate loss
    train_features, train_mp_adj, train_labels, train_row_idx, train_col_idx = dl.get_phase('train')
    val_features, val_mp_adj, val_labels, val_row_idx, val_col_idx = dl.get_phase('valid')

    # normalize features
    train_features, mean, std = dl.normalize_feature(train_features, return_moments=True)
    val_features = dl.normalize_feature(val_features, mean=mean, std=std, return_moments=False)

else:
    raise NotImplementedError('Dataloader for dataset {} is not supported yet!'.format(DATASET))

# convert features to tensors
train_features = torch.from_numpy(train_features).to(DEVICE)
val_features = torch.from_numpy(val_features).to(DEVICE)
train_labels = torch.from_numpy(train_labels).float().to(DEVICE)
val_labels = torch.from_numpy(val_labels).float().to(DEVICE)

train_row_idx = torch.from_numpy(train_row_idx).long().to(DEVICE)
train_col_idx = torch.from_numpy(train_col_idx).long().to(DEVICE)
val_row_idx = torch.from_numpy(val_row_idx).long().to(DEVICE)
val_col_idx = torch.from_numpy(val_col_idx).long().to(DEVICE)

# get support adjacency matrix [A0, ..., AS] 
train_support = compute_degree_support(train_mp_adj, DEGREE, adj_self_connections=ADJ_SELF_CONNECTIONS)
val_support = compute_degree_support(val_mp_adj, DEGREE, adj_self_connections=ADJ_SELF_CONNECTIONS)

# normalize these support adjacency matrices
for i in range(len(train_support)):
    train_support[i] = normalize_nonsym_adj(train_support[i])
    val_support[i] = normalize_nonsym_adj(val_support[i])
# convert to tensor
val_support = [csr_to_sparse_tensor(adj).to(DEVICE) for adj in val_support]

num_supports = len(train_support)

settings = {
    'num_support': num_supports,  
    'dropout': DROP,
    'batch_norm': BATCH_NORM,
    'learning_rate': LR,
    'wd': WD,
}


# create model
model = CompatibilityGAE(
    input_dim=train_features.shape[1],
    hidden=HIDDEN,
    num_classes=2,
    settings=settings
)
model.to(DEVICE)
model.train()

print("\nModel: ")
print(model)

best_val_acc = 0
best_epoch = 0
best_val_train_acc = 0
best_train_acc = 0 

for epoch in range(NB_EPOCHS):
    if SUP_DROP > 0:
        # do not modify the first support adj matrix, which is self-connections
        epoch_train_supports = []
        for i in range(1, len(train_support)):
            sampled_adj = support_dropout(train_support[i].copy(), SUP_DROP, drop_edge=True)
            # binarilize the support adjacency matrix
            sampled_adj.data[...] = 1 
            sampled_adj = normalize_nonsym_adj(sampled_adj)
            # convert adj sparse matrix to sparse tensor
            sampled_adj = csr_to_sparse_tensor(sampled_adj).to(DEVICE)
            # update message passing graph feed to the model
            epoch_train_supports.append(sampled_adj)
    else:
        epoch_train_supports = [csr_to_sparse_tensor(adj).to(DEVICE) for adj in train_support]
    # run one epoch
    train_avg_loss, train_acc = model.train_epoch(train_features, epoch_train_supports, train_row_idx, train_col_idx, train_labels)
    with torch.no_grad():
        model.eval()
        valid_pred = model.predict(val_features, val_support, val_row_idx, val_col_idx)
        valid_acc = model.accuracy(valid_pred, val_labels)
    model.train()

    print("Epoch: {} -- Train Loss: {:.4f} -- Train Acc: {:.4f} -- Valid Acc: {:.4f}".format(epoch, train_avg_loss, train_acc, valid_acc))

    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        best_epoch = epoch 
        best_val_train_acc = train_acc 
    
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    
print("Training Done!")
print("Best Epoch: {} -- Best Valid Acc: {:.4f} -- Best Train Acc at epoch {}: {:.4f} -- Best Overall Train Acc: {:.4f}".format(
    best_epoch, best_val_acc, best_epoch, best_val_train_acc, best_train_acc))
