# Cat-vs-Dog-Kaggle

<!-- TABLE OF CONTENTS -->
## TABLE OF CONTENTS
* [About the Project](#about-the-project)
* [Installation](#installation)
  * Git
    * for MacOS https://www.atlassian.com/git/tutorials/install-git#mac-os-x
    * for Windows https://www.atlassian.com/git/tutorials/install-git#windows
    * for Linux https://www.atlassian.com/git/tutorials/install-git#linux
  * Cloning the repository
  * Download Pycharm IDE
* [Development Summary](#Development-Summary)


<!-- ABOUT THE PROJECT -->
## About The Project
This project was created after completion of the Deep Learning by deeplearning.ai on Coursera specialization in order to apply the learned concepts and techniques from the Convolutional Neural Network course. [More details on development process here.](#development-summary)

<!-- Installation -->
## Installation
Git
* for MacOS https://www.atlassian.com/git/tutorials/install-git#mac-os-x
* for Windows https://www.atlassian.com/git/tutorials/install-git#windows
* for Linux https://www.atlassian.com/git/tutorials/install-git#linux

Cloning the repository
https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository

Download Pycharm IDE
https://www.jetbrains.com/pycharm/download/#section=mac

## Usage


## Development Summary 

This project began by constantly comparing and cross-referencing multiple sources in order to understand the different methods to structure the project, the different methods of implementing data preprocessing, and the reasons for deciding to use each of these methods. 

At first, the labeled training and unlabeled test images were preprocessed into 64x64 px images and saved to a train.npy and test.npy file respectively. This was done in order to perform numpy array slicing to split the training data into separate training, validation, and test sets. From models 0 through 16, the primary improvements that were made to the project workflow include the addition of a model performance plotting function, which was used to analyze bias and variance using training and validation accuracy and loss; and automatic saving for trained models and model history, which was used alongside the plots to analyze the performance of new model iterations. 

In terms of accuracy and loss, models 0 to 7 focused on reducing bias. Model 7 had a bias of 1.41% and a variance of 20.96%.
![a](/model_7_accuracy.png) 

Since both training accuracy and loss had begun to stagnate, models 8 and 9 focused on reducing variance. Batch normalization was added to model 9 which successfully reduced variance to 8.71% from 20.96% in model 7. (model 9 bias = 5.69%, variance = 8.71%)
![b](/model_9_accuracy.png) 

Models 10 to 13 experimented with zero padding and learning rate which resulted in a reduced variance of 2.75%. The main decrease to variance occurred in model 14 after the addition of a third dense layer which achieved 0.34% variance. (model 14 bias = 18.84%, variance = 0.34%)

![c](/model_14_accuracy.png)

Since the variance was low compared to the bias, model 15 added another conv2D block and tested the effects of same padding. The version without same padding and the additional conv2D block produced the expected result of decreased bias and increased variance from the parameter count increase. On the other hand, because same padding uses the edge pixels more often and maintains the input shape, less information was lost when propagating the input through each conv2D layer. Therefore, the version with same padding and the additional conv2D block resulted in lowered bias and maintained a low variance. (model 15 bias = 16.40%, variance = 0.44%). 
With Same Padding             |  Without Same Padding
:-------------------------:|:-------------------------:
![](/model_15_accuracy_wo_same_pad.png)  |  ![](/model_15_accuracy_w_same_pad.png)




To further reduce bias, the number of conv2D layers were duplicated which increased the model's ability to better fit the training set. (model 16 bias = 9.62%, variance = 7.08%)


After model 16, I discovered keras.callbacks and used csvlogger to auto log the epoch, accuracy, loss, validation accuracy, and validation loss. However, csvlogger did not save the time taken for each epoch and each step like the result of `model.summary`. Therefore, along with auto logging, I implemented summary saving, and automatic loss and accuracy plot saving.

At this point, each epoch of the 20 epochs took around 227 seconds which corresponded to a total of 4540 seconds which is about 75.67 minutes. Therefore, I studied TensorFlows use with a GPU (link here) and modified my workflow so that all implementation would be done on my Mac and all model training/testing would be done on my PC.




added modelcheckpoint after model 21

models 0 to 16 also showed promising headroom for further model improvement. For instance  
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification/notebook#Virtualize-Training
https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8

###### TLDR

- [ ] finish prog summary 
- [ ] include examples of model performance.
- [ ] change bias and variance measurements to accuracy instead of loss?
- [ ] format images `<img src="/model_15_accuracy_wo_same_pad.png" width="450">`
