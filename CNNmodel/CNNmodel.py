import utility_functions
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

def new_model(arch, h1_units, h2_units, output_units):
    '''
    Creates a pretrained model with 2 hidden layers.
    
    Inputs:
        arch - model architecture.
        input_units - the number of input units corresponding to the model architecture.
                        eg. vgg19 - 25088.
        h1_units - number of units in hidden layer 1.
        h2_units - number of units in hidden layer 2.
        output_units - number of units the model will output.
                        
    Outputs:
        model - it outputs the new pretrained model.
    '''
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad=False
    classifier = nn.Sequential(nn.Linear(25088, h1_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(h1_units, h2_units),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(h2_units, output_units),
                            nn.LogSoftmax(dim=1)
                                            )
    model.classifier = classifier
    return model

def train(model, epochs, trainloader, validloader, gpu, criterion, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)
    
    criterion = criterion
    optimizer = optimizer
    steps = 0

    for e in range(epochs):
        running_loss = 0    
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            optimizer.zero_grad()
        
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % 25 == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                
                        log_ps = model(images)
                        batch_loss = criterion(log_ps, labels)
                        validation_loss += batch_loss.item()
                
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
        
                print('Epoch: ' + str(e),
                      'Train Error: ' + str(running_loss / steps),
                      'Validation Error: ' + str(validation_loss / len(validloader)),
                      'Accuracy: ' + str(accuracy.item() / len(validloader) * 100) + '%')
                model.train()
        steps = 0
        
def save_checkpoint(model, arch, epochs, criterion, optimizer, class_to_idx, save_dir):
    checkpoint = {'classifier' : model.classifier,
              'arch': arch,
              'epochs': epochs,
              'criterion': criterion,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': class_to_idx}
    torch.save(checkpoint, save_dir)

def load_checkpoint(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    return model

def predict(model, image_path, topk, labels, gpu):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Inputs:
        topk - number of top values we want to retrieve.
    Outputs:
        top_probs - 
        labels_list - 
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = utility_functions.process_image(image_path)
        image = torch.from_numpy(np.array([image])).float()
        image.to(device)
        log_ps = model(image)
        ps = torch.exp(log_ps)
        probs, classes = ps.topk(topk, dim=1)
        
        top_probs = probs.tolist()[0]
        top_classes = classes.tolist()[0]
        idx_to_class = {value:key for key, value in model.class_to_idx.items()}
        labels_list = []
        for c in top_classes:
            labels_list.append(labels[idx_to_class[c]])
            
        utility_functions.imshow(utility_functions.process_image(image_path))

        predictions, classes = top_probs, labels_list
        fig, ax = plt.subplots()
        y = np.arange(len(classes))
        plt.barh(y, predictions, align='center')
        plt.yticks(np.arange(len(classes)), classes)
        plt.show()
    
        return top_probs, labels_list
    
def sanity_checking(model, image_path, index, labels):
    utility_functions.imshow(utility_functions.process_image(image_path)).set_title(labels[str(index)])

    predictions, classes = predict(image_path, model)
    fig, ax = plt.subplots()
    y = np.arange(len(classes))
    plt.barh(y, predictions, align='center')
    plt.yticks(np.arange(len(classes)), classes)
    plt.show()