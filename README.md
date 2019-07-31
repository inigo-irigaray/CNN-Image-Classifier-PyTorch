# CNN Image Classifier PyTorch

This project aims to build a command line application that can be trained on any set of labeled images. It has been broken down in 2 parts. The first part is the implementation on a jupyter notebook of the VGGnet architecture on a dataset of flower images to generate the code for a deep learning image classifier ([Classifier Development](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/tree/master/Classifier-Development)). The second part is the generalization of the code to create a convolutional neural network that allows the user to create his/her own classifier on any image dataset ([CNN model](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/tree/master/CNNmodel)).


**1. Loading, preprocessing and visualizing the dataset.** The first step was developing a the functions to load, pre-process and display the data. It can be helpful for the user to visualize data beforehand to familiarize himself/herself with the set. (These functions can be found on [utility-functions.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/utility_functions.py))

**2. Generating the new neural network model.** Secondly, I created a function that generates a pretrained model with two hidden layers and an output layer size equal to the number of possible classification categories. (These functions can be found on [CNNmodel.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/CNNmodel.py))

**3. Training the image classifier on the dataset.** Thirdly, the training dataset is processed through the pipeline to train the neural network. (These functions can be found on [train.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/train.py) and [CNNmodel.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/CNNmodel.py))

**4. Predicting the category of each image on the dataset.** Finally, the user can process the image he/she wants to classify through the neural network and it will output the most likely category with a high confidence level. (These functions can be found on [predict.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/predict.py.py) and [CNNmodel.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/CNNmodel.py))

## User inputs:

**Train function**

`--data_dir`: set the directory of your own training data.

`--batch_size`: set the size of the batches.

`--arch`: choose between using a vgg19 or a densenet121 architecture for your neural network.

`--h1_units`: set the size of the first hidden layer.

`--h2_units`: set the size of the second hidden layer.

`--json_file`: use your own label data for the training and classification.

`--learning_rate`: set your learning rate for the backpropagation.

`--epochs`: set the number of epochs to train the neural network.

`--gpu`: decide whether you wan to process the algorithm on the gpu (if available) or the cpu.

`--save_dir`: set the directory where you want to save the checkpoint.

**Predic function**

`--image_path`: decide which image you want the CNN to predict by giving its filepath.

`--checkpoint_file`: import the checkpoint you saved after training.

`--json_file`: use your own label data for the training and classification.

`--topk`: set the number of top values you want to retrieve.

`--gpu`: decide whether you wan to process the algorithm on the gpu (if available) or the cpu.

---------

# 1. Prerequisites.

- PyTorch
- OpenCV
- NumPy
- MatPlotLib
- PIL/pillow

# 2. Loading, preprocessing and visualizing the dataset.

In this first step, I created functions to load the data from the directories; to resize, crop, transform, normalize and classify the data into training, validation and testing datasets using `torchvision`; and to display the images.

The dataset need to resize to 224x224, since the vgg network takes in images of that size. Additional transformations on the training set, e.g. rotation or flipping, enables a better training on a more diverse set. The validation and testing sets measure the model's performance on images it hasn't processed yet.

Finally, the images need to be normalized, with values for the means [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.

<p align="center"> <img src="https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/Classifier-Development/preprocessing-display-example.png" height=300 width=350> <p/>

# 3. Generating the new neural network model.

In this next step, I created a function that generates a pretrained model with two possible architectures for the user to choose: vgg19 and densenet121, with former set by default. This model takes an input layer size of 25,088, with two hidden layers (the user can freely set their size) and an output layer size equal to the number of possible classification categories. This model is created with the ReLU activation function and the LogSoftMax function, with a dropout of p=0.2.

# 4. Training the image classifier on the dataset.

For training, I created a model that uses Negative Log-Likeliood Loss and the Adam optimization, since they allow for quicker training of neural networks, efficient and correct learning. The model lets the user set the epochs, with a default of 6. It is important to note that in general the higher the number of epochs, the more accurate the model will get, since it will repeat the feed-forward-backpropagation process more times. 

After training, the model trained with the user's dataset is then saved in his/her directory of choice as a checkpoint file (.pth).

# 5. Predicting the category of each image on the dataset.

Finally, 

   <p align="center"> <img src="https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/Classifier-Development/predict-example.png" height = 450 width=400 > <p/>


# 6. How to use?

For use, the repository should be cloned or downloaded. Running the main-classifier.py on the Terminal processes the image dataset given. To use the algorithms on other images the user must make a minor change in main-classifier.py in the variable that keeps the directory of the files to be classified. And then, the program should run smoothly and deliver the expected results.

