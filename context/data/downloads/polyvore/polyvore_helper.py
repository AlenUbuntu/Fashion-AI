import os 
import json 
import argparse 
import pickle
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import numpy as np 

parser = argparse.ArgumentParser(description='preprocess polyvore dataset - get image features and graph adjacency matrix')

parser.add_argument('--phase', default='train', choices=['train', 'valid', 'test'])
parser.add_argument('--root', default='/home/alan/Downloads/fashion/polyvore', type=str)
parser.add_argument('--save-path', type=str, default='dataset')
parser.add_argument('--d', type=int, default=1024, help='dimension of image feature vector extracted by a pre-trained CNN model.')

args = parser.parse_args()

# create output path 
output_path = os.path.join(args.root, args.save_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# load json
json_path = os.path.join(args.root, 'jsons', '{}_no_dup.json'.format(args.phase))
with open(json_path, 'r') as f:
    json_data = json.load(f)

# load pre-processed image features
# it is the results obtained with 'extract_feature.py'
feat_pkl_path = os.path.join(os.path.join(args.root, args.save_path), 'imgs_featdict_{}.pkl'.format(args.phase))
if os.path.exists(feat_pkl_path):
    with open(feat_pkl_path, 'rb') as f:
        image_feat = pickle.load(f)
else:
    raise FileNotFoundError('The image feature file {} does not exist!'.format(feat_pkl_path))

# record edges of outfit graph
edges = {}
# map original id in url to another continuously increasing id
id2idx = {}
# map the url id of an item to its formated id - "OUTFIT-ID_INDEX", where INDEX counts within a outfit.
id2infoid = {}

# start idx 
idx = 0
# initialize feature vector holder
features = []

# note that the same item may participate in multiple outfits
# if node A and node B are two items that once appear in the same outfit, they are connected
# In other words, node A is connected to other nodes that once appear together with it in any outfit.
# node B, node C may both connect to node A. but (node A, node B) is from outfit 1, (node A, node C) 
# is from outfit 2
for outfit in json_data:
    outfit_ids = set()

    # find url ids of items in an outfit
    # maps url ids to their formated id OUTFIT-ID_INDEX
    for item in outfit['items']:
        # get id from image url 
        _, id_ = item['image'].split('id=')
        id_ = int(id_)
        outfit_ids.add(id_)
        id2infoid[id_] = '{}_{}'.format(outfit['set_id'], item['index'])

    # find edge info
    for id_ in outfit_ids:
        if id_ not in edges:
            edges[id_] = set()  # empty set 

            # find corresponding image feature and add to features
            # id_ not in edges ensure we only visit each node once
            img_feature = image_feat[str(id_)]
            features.append(img_feature)

            # map this id to an sequential growing id
            id2idx[id_] = idx
            # update indexer
            idx += 1
    
        # for each item in an outfit, we update its neighbor info 
        edges[id_].update(outfit_ids)
        # remove id_ itself to avoid self loop
        edges[id_].remove(id_)

# save mapping to files
save_path = os.path.join(args.root, args.save_path, 'id2idx_{}.json'.format(args.phase))
with open(save_path, 'w') as f:
    json.dump(id2idx, f)
save_path = os.path.join(args.root, args.save_path, 'id2InfoId_{}.json'.format(args.phase))
with open(save_path, 'w') as f:
    json.dump(id2infoid, f)

# create graph adjacency matrix
# idx - n (0 - n -1)
adj = lil_matrix((idx, idx))
# create feature matrix
x = np.zeros((idx, args.d))

print("Processing graph node feature matrix and graph adjacency matrix ... ")

for id_ in edges:
    start_idx = id2idx[id_]

    rel_ids = edges[id_]
    x[start_idx] = features[start_idx]

    for to_id in rel_ids:
        to_idx = id2idx[to_id]

        # filling entries in the adjacency matrix
        adj[start_idx, to_idx] = 1
        # we consider graph is undirectional, A->B and B->A since A and B are both in a outfit.
        adj[to_idx, start_idx] = 1 

print("Done !")

# save the adjacency matrix
save_path = os.path.join(args.root, args.save_path, 'adj_{}.npz'.format(args.phase))
adj = adj.tocsr()
save_npz(save_path, adj)

# save feature matrix
save_path = os.path.join(args.root, args.save_path, 'X_{}.npz'.format(args.phase))
x = csr_matrix(x)
save_npz(save_path, x)



