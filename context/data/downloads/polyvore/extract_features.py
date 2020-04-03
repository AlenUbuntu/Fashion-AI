"""
Extract features for each item image, using a resnet50 with PyTorch
"""
import torch 
import torch.nn as nn
import argparse 
import os
import json
import pickle
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


def extract_feature(transform, image):
    pass

parser = argparse.ArgumentParser(description='Image feature extractor')
parser.add_argument('--phase', default='train', choices=['train', 'valid', 'test'])
parser.add_argument('--root', default='/home/alan/Downloads/fashion/polyvore', type=str)
parser.add_argument('--save-path', type=str, default='dataset')
args = parser.parse_args()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# load resnet to extract features
model = models.resnet50(pretrained=True)
# remove the last layer - classifier 
# shallower network may be better
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)

# build transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


image_folder = os.path.join(args.root, 'images')
json_file = os.path.join(args.root, 'jsons', '{}_no_dup.json'.format(args.phase))

# create save path 
save_dir = os.path.join(args.root, args.save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'imgs_featdict_{}.pkl'.format(args.phase))

# read data
with open(json_file) as f:
    json_data = json.load(f)

# url id is unique, outfit_id is also unique
# however, an item may appear in multiple outfits.
# It means that outfit_id stored in urlId2outfitId is the last outfit 
# where an item appears.
# Note that an item may have multiple outfit_id, but only one url id
urlId2outfitId = {}
for outfit in json_data:
    for item in outfit['items']:
        # get url id
        _, id_ = item['image'].split('id=')
        outfit_id = '{}_{}'.format(outfit['set_id'], item['index'])
        urlId2outfitId[id_] = outfit_id 

# initialize empty feature dict
id2feature = {}

with torch.no_grad():
    for i, id_ in enumerate(urlId2outfitId):
        set_id, index = urlId2outfitId[id_].split('_')

        # construct image path 
        img_path = os.path.join(args.root, 'images', set_id, '{}.jpg'.format(index))

        # check if the path exists
        assert os.path.exists(img_path)

        # read image
        img = Image.open(img_path)
        # convert to RGB format
        img = img.convert('RGB')

        # extract feature
        img = transform(img)
        img = img.unsqueeze(0).to(device)
        feature = model(img).squeeze().cpu().numpy()

        # store
        id2feature[id_] = feature

        print("Process: {}/{}".format(i+1, len(urlId2outfitId)), end='\r')

# save 
with open(save_path, 'wb') as f:
    pickle.dump(id2feature, f)