import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import utility_functions
import CNNmodel
import argparse

parser = argparse.ArgumentParser(description= 'Prediction of the image label')
parser.add_argument('image_path', help = 'Path to the images to predict')
parser.add_argument('checkpoint_file', help = 'Path to the saved neural network.')
parser.add_argument('--json_file', default = './cat_to_name.json', help = 'Path to json file storing category labels.')
parser.add_argument('--topk', type = int, default = 5, help = 'Number of top values we want to retrieve.')
parser.add_argument('--gpu', type = bool, action = 'store_true', default = False, help = 'Use if you want to train on GPU.')

arguments = parser.parse_args()
image_path = arguments.image_path
checkpoint_file = arguments.checkpoint_file
json_file = arguments.json_file
topk = arguments.topk
gpu = arguments.gpu

model = CNNmodel.load_checkpoint(checkpoint_file)
labels = utility_functions.json_loader(json_file)

CNNmodel.predict(model, image_path, topk, labels, gpu)