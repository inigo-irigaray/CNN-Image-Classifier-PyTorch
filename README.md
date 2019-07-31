# CNN Image Classifier PyTorch

This project aims to build a command line application that can be trained on any set of labeled images. It has been broken down in 2 parts. The first part is the implementation on a jupyter notebook of the VGGnet architecture on a dataset of flower images to generate the code for a deep learning image classifier ([Classifier Development](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/tree/master/Classifier-Development)). The second part is the generalization of the code to create a convolutional neural network that allows the user to create his/her own classifier on any image dataset ([CNN model](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/tree/master/CNNmodel)).


**1. Loading, preprocessing and visualizing the dataset.** The first step was developing a the functions to load, pre-process and display the data. It can be helpful for the user to visualize data beforehand to familiarize himself/herself with the set. (These functions can be found on [utility-functions.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/utility_functions.py))

**2. Training the image classifier on the dataset.** Secondly, the training dataset is processed through the pipeline to train the neural network. (These functions can be found on [train.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/train.py) and [CNNmodel.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/CNNmodel.py))

**3. Predicting the category of each image on the dataset.** Finally, the user can process the image he/she wants to classify through the neural network and it will output the most likely category with a high confidence level. (These functions can be found on [predict.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/predict.py.py) and [CNNmodel.py](https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/CNNmodel/CNNmodel.py))

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

# 2. Loading and visualizing the data.

This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:

* 904 red traffic light images
* 536 green traffic light images
* 44 yellow traffic light images

<p align="center"> <img src="https://github.com/inigo-irigaray/CNN-Image-Classifier-PyTorch/blob/master/Classifier-Development/preprocessing-display-example.png" height=300 width=350> <p/>

# 3. Pre-processing.

For classification tasks like this one we need to create features by performing the same analysis on different pictures. It is, therefore, important that similar images create similar features. And to facilitate this, we will standardize the input and output to understand what results we can expect from running the program.

First, I created a function that takes an image and resizes it to 32x32. Square images can be rotated and analyzed in smaller square patches. Additionally, if all images are the same size we can confidently pass them through the same pipeline.

Secondly, it is customary to convert categorical labels like 'red' to numerical values. I created a function that one-hot encodes the labels into a 1D list of zeros with a number one representing the categorical value in the following way:
  - red     =   [1,0,0]
  - yellow  =   [0,1,0]
  - green   =   [0,0,1]

Finally, I created a function to standardize a list of images and pair each image to its one-hot encoded label.

# 4. Feature extraction.

In this part of the project I thought about the different features and layers that combine to form a colored image and which ones are more characteristic of the areas that would represent the red, yellow and green lights in the image. I converted the images to HSV for the whole process.

I thought that it would be best to apply first a saturation mask to identify the colorful areas in the image, independently of illumination, which differed in some images and can be affected by weather conditions for instance. Since I expected the color light to be one of the most saturated parts of the image, I applied a minimum threshold that is the average saturation plus the standard deviation of the saturation of the image. Thus, I make sure that I just take into account the parts of the image that are systematically saturated above average and account for differences from image to image.

Secondly, I applied a brightness mask on the saturation-masked image, since an illuminated traffic light can be expected t be brighter than the rest of the image. I applied a similar reasoning when writing the minimum threshold as in the saturation mask. This time, instead of adding the standard deviation to the mean of the brightness, I added a scalar times the standard deviation, as I expected the illuminated part of the traffic light would much brighter than the rest of the masked image. After some testing I found a scalar of 2 to be the best fit for the algorithm.

Finally, I created three color spacing functions to count the number of red, yellow and green pixels in the brightness-and-saturation-masked image. Here I found the green thresholds to be one of the most sensitive to changes. I, therefore, had to make multiple tests to fine-tune the thresholds. The yellow one was more receptive to changes and didn't affect the overall performance of the algorithm that much. Overall, it was similar with the red thresholds. However, here I found that it was essential when aiming for virtually 100% accuracy. That is why I created two models, depending on the red threshold.

   - In the first one, I managed to achieve a **~99% accuracy**, with a lower bound of 120 in the red count function, with only 3 out of 297 misclassifications in the testing set. However, one of those was a red light classify as a green light, which is very unsafe for driving. To be fair, it is a very low-quality image, that makes the bottom part of the image look greenish and the red is yellowish, as you can see below. To solve this issue I tried applying image-localization filters for the colors, brightness and saturation features, it is a general assumption that red lights are on top and green lights at the bottom. I also made multiple adjustments to thresholding in the existing functions. However, I never managed to keep or increase the accuracy while eliminating the dangerous misclassification.

   <p align="center"> <img src="https://github.com/inigo-irigaray/Traffic-Light-Classifier-Symbolic-AI/blob/master/Traffic-Light-99%25/images/red-misclass-green.png" height = 300 width=250 > <p/>

   - In the second one, I adjusted the red lower bound to 107, to increase the red spectrum and ensure a higher count of pixels in the image. **My accuracy decreases to ~97%**, but I eliminated the unwanted red as green misclassification.

# 5. How to use?

For use, the repository should be cloned or downloaded. Running the main-classifier.py on the Terminal processes the image dataset given. To use the algorithms on other images the user must make a minor change in main-classifier.py in the variable that keeps the directory of the files to be classified. And then, the program should run smoothly and deliver the expected results.

# 6. License.

All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
<p float="left">
