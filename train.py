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

parser = argparse.ArgumentParser(description= 'Training the neural network.')
parser.add_argument('data_dir', default = './flowers', help = 'Path to the images to train, validate and test the neural network.')
parser.add_argument('--batch_size', type = int, default = 64, help = 'Size of training batches.')
parser.add_argument('--arch', choices = ['vgg19', 'densenet121'], default = 'vgg19', help = 'Model architecture. Takes vgg19 or densenet121 as inputs.')
parser.add_argument('--h1_units', type = int, default = 510, help = 'The size of the first hidden layer.')
parser.add_argument('--h2_units', type = int, default = 204, help = 'The size of the second hidden layer.')
parser.add_argument('--json_file', default = './cat_to_name.json', help = 'Path to json file storing category labels.')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate used for the model.')
parser.add_argument('--epochs', type = int, default = 6, help = 'The number of epochs for training.')
parser.add_argument('--gpu', type = bool, action = 'store_true', default = False, help = 'Use if you want to train on GPU.')
parser.add_argument('--save_dir', default = './checkpoint.pth', help = 'The path to store the neural network checkpoint.')

arguments = parser.parse_args()
data_dir = arguments.data_dir
batch_size = arguments.batch_size
arch = arguments.arch
h1_units = arguments.h1_units
h2_units = arguments.h2_units
json_file = arguments.json_file
learning_rate = arguments.learning_rate
epochs = arguments.epochs
gpu = arguments.gpu
save_dir = arguments.save_dir

trainloader, validloader, testloader, train_data = utility_functions.data_loader(data_dir, batch_size)
output_units = len(utility_functions.json_loader(json_file))
model = CNNmodel.new_model(arch, h1_units, h2_units, output_units)

if model != 0:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    class_to_idx = train_data.class_to_idx
    CNNmodel.train(model, epochs, trainloader, validloader, gpu, criterion, optimizer)
    CNNmodel.save_checkpoint(model, arch, epochs, criterion, optimizer, class_to_idx, save_dir)