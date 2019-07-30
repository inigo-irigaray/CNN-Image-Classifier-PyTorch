import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

def data_loader(data_dir, batch_size):
    '''
    It loads, transforms and separates all the data from the directory into training,
    validation and test datasets.
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomRotation(25),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         [0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                                     ])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         [0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                                     ])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=False)
    
    return trainloader, validloader, testloader, train_data

def json_loader(json_file):
    '''
    It loads a JSON file that maps the class values to other category names.
    '''
    with open(json_file, 'r') as f:
        return json.load(f)
    
def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array.
    '''
    pil = Image.open(image)
    pil = pil.resize((256,256))
    pil = pil.crop((16,16,240,240))
    pil = np.array(pil) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pil = (pil - mean)/std
    pil = pil.transpose((2,0,1))
    return pil
    
def imshow(image):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax