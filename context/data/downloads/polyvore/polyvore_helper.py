import os 
import json 
import argparse 
import pickle
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import numpy as np 
import random 

"""
xxx_no_dup.json
[
    {
        'name': xxx,
        'views': xxx,
        'items': [
            {
                "index": 1,
                "name": xxx,
                "price": xxx,
                "likes": xxx,
                "image": xxx,
                "categoryid": xxx
            },
            ...
        ],
        'image': xxx,
        'likes': xxx,
        'date': xxx,
        'set_url': xxx,
        'set_id': xxx,
        'desc': xxx
    },
    ...
]

fill_in_blank_test.json
[
    {
        "question":[  # SET-ID_INDEX from same SET
            xxx,  
            xxx,  
            xxx,
            xxx
        ],
        "answers":[   # SET-ID_INDEX from different sets, one is the true answer
            xxx,
            xxx,
            xxx,
            xxx
        ],
        'blank_position': x
    },
    ...
]
"""

def resample_fill_the_blank(data_dir):
    """
    Resample the FITB choices so that they share category with the correct item.
    """
    print('Start resampling of fill-the-blank ...')
    question_file = os.path.join(data_dir, 'fill_in_blank_test.json')
    json_file = os.path.join(data_dir, '{}_no_dup.json'.format('test'))
    question_file_resampled = os.path.join(
        data_dir, 'fill_in_blank_test_RESAMPLED.json')

    # read json data
    with open(json_file) as f:
        json_data = json.load(f)
    
    with open(question_file) as f:
        question_data = json.load(f)
    
    questions_resampled = []

    print("Number of questions: {}".format(len(question_data)))
    print("Number of test outfits: {}".format(len(json_data)))

    not_enough = 0

    outfitId2cat = {}
    for outfit in json_data:
        for item in outfit['item']:
            key = '{}_{}'.format(outfit['set_id'], item['index'])   # OUTFIT-ID_INDEX
            outfitId2cat[key] = item['categoryid']
        
    for ques in question_data:
        new_ques = {
            'blank_position': ques['blank_position']
        }
        q = []
        for q_id in ques['question']:  # q_id: OUTFIT-ID_INDEX
            q.append(q_id)
        
        set_id = q_id.split('_')[0]  # all q_ids in a question belong to the same outfit

        new_ques['question'] = q 

        ans = []

        # resample answers so that all or most items are from the same category - looks similar
        # This is a kind of hard mining technique, it increase the difficulty of the
        # question.
        for i, ans_id in enumerate(ques['answers']):
            if i == 0:  # correct choice
                assert ans_id.split('_')[0] == set_id
                cat_id = outfitId2cat[ans_id]
                # find all possible items of the category same as the correct choice but are neither not the
                # answer itself nor the question items
                choices = set([id_ for id_ in outfitId2cat if outfitId2cat[id_] == cat_id]) - set([ans_id]) - set(q)

                # random shuffle the choices
                choices = list(choices)
                random.shuffle(choices)
            else:
                # resample item that has the category 'cat_id' (which is the same as the missing item)
                # it could happen that there aren't enough items of that category
                # note that choices has removed the correct choice itself, thus, i corresponds to i-1
                # in choices. 
                if i - 1 < len(choices):
                    ans_id = choices[i-1]
                else:
                    # not enough candidate items in this category
                    # randomly select one item from all items
                    ans_id = random.choice(list(outfitId2cat.keys()))
                    not_enough += 1
            
            ans.append(ans_id)

        new_ques['answers'] = ans 

        questions_resampled.append(new_ques)
    
    print('Not enough: ', not_enough)

    # save resampled questions
    with open(question_file_resampled, 'w') as f:
        json.dump(questions_resampled, f)
    
    print("Resampling of fill-the-blank is done.")


def resample_compatibility(data_dir):
    """
    The invalid outfits for the compatibility prediction task are very easy, since they don't need to form a valid 
    outfit. For example, we could have a cotton-packed jacket and a short appearing in the same outfit. That's why 
    we need to resample the invalid outfits, so that they have the same categories as the valid ones, making them 
    be invalid only because of their styles.
    """
    print('Start resampling of compatibility ...')
    orig_file = os.path.join(data_dir, 'fashion_compatibility_prediction.txt')
    new_file = os.path.join(
        data_dir, 'fashion_compatibility_prediction_RESAMPLED.txt')
    
    # find valid and invalid outfits
    valid_outfits = []
    invalid_outfits = []
    with open(orig_file, 'r') as f:
        for line in f:
            res = line.strip().split(' ')
            label, items = int(res[0]), res[1:]
            if label == 1:
                valid_outfits.append(items)
            else:
                invalid_outfits.append(items)
    
    # read files
    with open(os.path.join(data_dir, 'test_no_dup.json')) as f:
        json_data = json.load(f)
    
    item2cat = {}
    cat_items = {}

    for outfit in json_data:
        set_id = outfit['set_id']
        for item in outfit['items']:
            index = item['index']
            id_ = '{}_{}'.format(set_id, index)
            item2cat[id_] = item['categoryid']

            if id_ not in cat_items:
                cat_items[item['categoryid']] = []
            
            cat_items[item['categoryid']].append(id_)
    
    # initialize new collections
    new_invalid = []
    
    for _ in range(len(invalid_outfits)):
        # sample an outfit from valid outfits, we are going to use its category combination
        base = random.choice(valid_outfits)

        new_outfit = []

        for item in base:
            item_cat = item['categoryid']
            # randomly sample another item from item_cat
            new_item = random.choice(cat_items[item_cat])
            # add to new outfit
            new_outfit.append(new_item)
        
        # add new outfit to new invalid collection
        new_invalid.append(new_outfit)
    
    # write the re-sampled outfits into a file
    with open(new_file, 'w') as f:
        
        # write valid outfits
        for outfit in valid_outfits:
            line = ' '.join(['1']+outfit)
            f.write(line+'\n')
        
        # write invalid outfits
        for outfit in new_invalid:
            line = ' '.join(['0']+outfit)
            f.write(line+'\n')
    
    print('Resampling of compatibility is done!')


def get_questions(data_dir, resample=False):
    if resample:
        question_file = os.path.join(
            data_dir, 'fill_in_blank_test_RESAMPLED.json')
    else:
        question_file = os.path.join(data_dir, 'fill_in_blank_test.json')
    
    json_file = os.path.join(data_dir, 'test_no_dup.json')

    # read files
    with open(json_file) as f:
        json_data = json.load(f)
    
    with open(question_file) as f:
        question_data = json.load(f)
    
    questions = []

    outfitId2Urlid = {}

    for outfit in json_data:
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], item['index'])
            # get url id
            _, id_ = item['image'].split('id=')
            outfitId2Urlid[outfit_id] = id_ 
    
    for ques in question_data:
        q = []
        for q_id in ques['question']:  
            # q_id = OutfitID-Index, map to url id
            q_id = outfitId2Urlid[q_id]
            q.append(q_id)
        
        set_id = ques['question'][0].split('_')[0]

        a = []
        blank_positions = []

        for i, a_id in enumerate(ques['answers']):
            if i == 0:
                # correct choice
                assert a_id.split('_')[0] == set_id 
            pos = int(a_id.split('_')[1])
            a_id = outfitId2Urlid[a_id]
            a.append(a_id)
            blank_positions.append(pos)
        
        # questions is a list of questions
        # each question is a list that contains
        #    - List of url ids of question items (len N)
        #    - List of url ids of possible answers (len 4)
        #    - List of positions where these answers corresponds to (len 4)
        #    - desired blank position
        questions.append([q, a, blank_positions, ques['blank_position']])
    
    return questions

def get_compatibility(data_dir, resample=False):
    """
    convert OutfitId to url id
    """
    json_file = os.path.join(data_dir, 'test_no_dup.json')
    with open(json_file) as f:
        json_data = json.load(f)
    
    outfitId2Urlid = {}

    for outfit in json_data:
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], item['index'])
            _, id_ = item['image'].split('id=')
            outfitId2Urlid[outfit_id] = id_
    
    if resample:
        compat_file = os.path.join(
            data_dir, 'fashion_compatibility_prediction_RESAMPLED.txt')
    else:
        compat_file = os.path.join(
            data_dir, 'fashion_compatibility_prediction.txt')
    
    outfits = []
    with open(compat_file) as f:
        for line in f:
            res = line.strip().split(' ')
            label = int(res[0])
            assert label in (0, 1)
            items = [outfitId2Urlid[outfit_id] for outfit_id in res[1:]]
            outfits.append((items, label))
    
    return outfits

def create_test(args, data_dir, id2idx, resample=False):
    if resample:
        resample_fill_the_blank(data_dir)
        resample_compatibility(data_dir)
    
    # create fill_in_the_blank data
    questions = get_questions(data_dir, resample=resample)

    for i in range(len(questions)):
        assert len(questions[i]) == 4, 'Expect a question data contains 4 lists, but got {}'.format(len(questions))

        # convert the url id in question and answer items to indecies
        for j in range(2):
            # j = 0 - question items
            # j = 1 - answer items
            for k in range(len(questions[i][j])):
                questions[i][j][k] = id2idx[questions[i][j][k]]
        
    question_file = os.path.join(args.root, args.save_path, 'questions_{}.json'.format(args.phase))
    if resample:
        question_file = question_file.replace(
            'questions', 'questions_RESAMPLED')
    
    with open(question_file, 'w') as f:
        json.dump(questions, f)
    
    # create outfit compatibility prediction data
    outfits = get_compatibility(data_dir, resample=resample)
    for i in range(len(outfits)):  # (items, label)
        for j in range(len(outfits[i][0])):
            outfits[i][0][j] = id2idx[outfits[i][0][j]]
    
    compat_file = os.path.join(
        args.root, args.save_path, 'compatibility_{}.json'.format(args.phase))
    if resample:
        compat_file = compat_file.replace(
            'compatibility', 'compatibility_RESAMPLED')
    
    with open(compat_file, 'w') as f:
        json.dump(outfits, f)

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
urlId2outfitId = {}

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
        outfit_ids.add(id_)
        urlId2outfitId[id_] = '{}_{}'.format(outfit['set_id'], item['index'])

    # find edge info
    for id_ in outfit_ids:
        if id_ not in edges:
            edges[id_] = set()  # empty set 

            # find corresponding image feature and add to features
            # id_ not in edges ensure we only visit each node once
            img_feature = image_feat[id_]
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
save_path = os.path.join(args.root, args.save_path, 'urlId2outfitId_{}.json'.format(args.phase))
with open(save_path, 'w') as f:
    json.dump(urlId2outfitId, f)

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

if args.phase == 'test':
    create_test(args, os.path.join(args.root, 'jsons'), id2idx, resample=False)
    create_test(args, os.path.join(args.root, 'jsons'), id2idx, resample=True)