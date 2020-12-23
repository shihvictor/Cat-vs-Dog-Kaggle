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

In terms of accuracy and loss, models 0 to 7 focused on reducing bias. 
![asdf](/model_7_accuracy.png) 
![asdf](/model_7_loss.png)

Since both training accuracy and loss had begun to stagnate, models 8 and 9 focused on reducing variance. Batch normalization was added to model 9 which successfully reduced variance to 0.3430 from 1.268 in model 7. Models 10 to 13 experimented with zero padding and learning rate which resulted in a training loss of 0.3443 and a validation loss of 0.4426. The main decrease to variance occurred after the addition of a third dense layer in model 14, which saw a decrease from 0.0983 in model 13 (loss: 0.3443, val_loss: 0.4426) to 0.0132 in model 14 (loss: 0.4108, val_loss: 0.4240).

Since the variance was low compared to the bias, model 15 tested the effects of same padding which produced the expected result of lowering bias (loss: 0.3689) since more of the original input features are propagated through the network. Expanding from model 15, model 16 duplicated the number of conv2D layers which further reduced the bias from 0.3689 to 0.2295. 

added csvlogger after model 16
added modelcheckpoint after model 21

models 0 to 16 also showed promising headroom for further model improvement. For instance  
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification/notebook#Virtualize-Training
https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8

- [ ] finish prog summary 
- [ ] include examples of model performance.
- [ ] change bias and variance measurements to accuracy instead of loss?
