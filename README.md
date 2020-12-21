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
* [Project Progress](#project-progress)
  * Progress Summary
  * Model Progress

<!-- ABOUT THE PROJECT -->
## About The Project
This project was created after completion of the Deep Learning by deeplearning.ai on Coursera specialization in order to apply the learned concepts and techniques from the Convolutional Neural Network course.

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

## Project Progress
Progress Summary

This project began by constantly comparing and cross-referencing multiple sources in order to understand the different methods to structure the project, the different methods of implementing data preprocessing, and the reasons for deciding to use each of these methods. 

At first, the labeled training and unlabeled test images were preprocessed into 64x64 px images and saved to a train.npy and test.npy file respectively. This was done in order to perform numpy array slicing to split the training data into separate training, validation, and test sets. From models 0 through 16, the primary improvements that were made to the project workflow include the addition of a model performance plotting function, which was used to analyze bias and variance using training and validation accuracy and loss; and automatic saving for trained models and model history, which was used alongside the plots to analyze the performance of new model iterations. In terms of accuracy and loss , Models 0 to 16 also showed promising headroom for further model improvement. For instance. Even though establishing the overall organization of the project, understanding the organization of each file, and  adding functions such as `plot_Acc_And_Loss(history_dict, save=True)` accuracy, there models 0~16 were used to 
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification/notebook#Virtualize-Training
https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8

- [ ] finish prog summary 
- [ ] include examples of model performance.
