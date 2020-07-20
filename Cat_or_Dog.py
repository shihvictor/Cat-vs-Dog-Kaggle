import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import Model
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path

from model20 import model20



TRAIN_DIR = 'datasets/train'
TEST_DIR = 'datasets/test'
IMG_SIZE = 64
# print(os.listdir(TRAIN_DIR))
"""CHANGE THESE TO SWITCH BETWEEN TRAINING AND LOADING"""
NEW_MODEL = False
MODEL_NAME = 'model_20'


def label_img(img_name):
    """
    Extracts the label of an image and converts the label to a one-hot vector
    :param img_name: name of format label.index.jpg
    :return: one-hot
    """
    word_label = img_name.split('.')[-3]
    # conversion to one-hot array [cat,dog] [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]


# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]


def create_train_data():
    """
    Changes images in the TRAIN_DIR directory to a numpy array
    :return: X_train : array of shape (m, IMG_SIZE, IMG_SIZE, 3)
    Y_train : array of shape (m, 2)
    """
    X_train = []
    Y_train = []
    # i = 0
    for img_name in tqdm(os.listdir(TRAIN_DIR)):
        # label for ex i = y(i). One-hot vector.
        y = np.array(label_img(img_name))     # [] makes shape (1, 2) instead of (2,)
        # read and resize img for ex i = x(i)
        image = np.array(plt.imread(TRAIN_DIR+'/'+img_name))
        x = resize(image, [IMG_SIZE, IMG_SIZE, 3]) # resizes the dim of the image array to fit to CNN

        # plt.imshow(x, interpolation='nearest')
        # plt.show()
        # Save features and label
        X_train.append(x)
        Y_train.append(y)
        # i += 1
        # if i == 1000: break
    # Convert to numpy array, shuffle, and save
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # shuffling--> https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    rng_state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_train)
    np.save('datasets/X_train_data.npy', X_train)
    np.save('datasets/Y_train_data.npy', Y_train)
    return X_train, Y_train


def process_test_data():
    X_test = []
    # i = 0
    for img_name in tqdm(os.listdir(TEST_DIR)):
        # read and resize img for ex i = x(i)
        image = np.array(plt.imread(TEST_DIR + '/' + img_name))
        x = resize(image, [IMG_SIZE, IMG_SIZE, 3])  # resizes the dim of the image array to fit to CNN

        # plt.imshow(x, interpolation='nearest')
        # plt.show()
        # Save features and label
        X_test.append(x)
        # i += 1
        # if i == 5: break
    # Uncomment this later to randomize order of imgs.
    # shuffle(training_data)
    X_test = np.array(X_test)
    np.save('datasets/X_test_data.npy', X_test)
    return X_test


def model(input_shape):
    """
    Keras model
    :return: Keras model using Keras functional API https://keras.io/guides/functional_api/
    """
    # Input layer
    X_input = Input(input_shape)

    X = ZeroPadding2D(padding=(2, 2))(X_input)

    # Block1
    X = Conv2D(filters=32, kernel_size=(5, 5), strides=1, name='conv2D_0')(X)
    X = BatchNormalization(axis=3, name='bn_0')(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_0')(X)
    X = Dropout(rate=.2)(X)

    # Block2
    X = Conv2D(filters=64, kernel_size=(5, 5), strides=1, name='conv2D_1')(X)
    X = BatchNormalization(axis=3, name='bn_1')(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(X)
    X = Dropout(rate=.2)(X)

    # Block3
    X = Conv2D(filters=128, kernel_size=(5, 5), strides=1, name='conv2D_2')(X)
    X = BatchNormalization(axis=3, name='bn_2')(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(X)
    X = Dropout(rate=.2)(X)

    # Flatten output of ReLU block
    X = Flatten()(X)

    # Fully connected layer with softmax activation
    X = Dense(units=256, activation="relu", name='fc_0')(X)
    X = Dropout(rate=.5)(X)
    X = Dense(units=2, activation="softmax", name='fc_1')(X)

    # Keras model instance
    model = Model(inputs=X_input, outputs=X, name='CatDogModel')
    return model


def plot_Acc_And_Loss(history_dict, save=True):
    """
    Plots loss and accuracy of train and val data over epochs.
    :return:
    """
    plt.plot(history_dict['accuracy'])
    plt.plot(history_dict['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save: plt.savefig('model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+"_accuracy.png")
    plt.show()

    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save: plt.savefig('model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+"_loss.png")
    plt.show()


""" LOAD DATA """
# === Either have (1) or (2) commented.
# X_data, Y_data = create_train_data()  # (1)
process_test_data()
# === If train data .npy file already created, load it instead.
X_data = np.load('datasets/X_train_data.npy')   #(2)
Y_data = np.load('datasets/Y_train_data.npy')   #(2)
X_test_data = np.load('datasets/X_test_data.npy')

""" PARTITION DATA """
M = X_data.shape[0]
X_train = X_data[0:int(M*.8), :]    # Train and cv data
Y_train = Y_data[0:int(M*.8), :]
X_val = X_data[int(M*.8):, :]
Y_val = Y_data[int(M*.8):, :]

print("number of training examples : " + str(X_train.shape[0]))
print("number of val examples : " + str(X_val.shape[0]))
print("X_train shape : " + str(X_train.shape))
print("Y_train shape : " + str(Y_train.shape))
print("X_val shape : " + str(X_val.shape))
print("Y_val shape : " + str(Y_val.shape))
print("X_test_data shape : " + str(X_test_data.shape))


# Compile, Train, Save, Plot
if NEW_MODEL:
    """Compile Model"""
    # cat_dog_model = model(X_train.shape[1:])
    cat_dog_model = model20(X_train.shape[1:])      # Create model
    cat_dog_model.summary()     # Print summary
    opt = Adam(learning_rate=.0001)     # Set optimizer
    cat_dog_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])     # Compile model

    """Train the model"""
    Path('model_logs/'+MODEL_NAME+'_logs/').mkdir(parents=True)
    csv_logger = CSVLogger(filename='model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+'_log.csv', separator=',', append=True)
    model_history = cat_dog_model.fit(x=X_train, y=Y_train, batch_size=32, epochs=30, validation_data=(X_val, Y_val), shuffle=True, callbacks=[csv_logger])
    #=== save the model ===
    cat_dog_model.save(filepath='model/'+MODEL_NAME, overwrite=True)

    """Save model history and plot loss and acc"""
    with open('model/'+MODEL_NAME+'/trainHistoryDict', 'wb') as file_name:
        pickle.dump(model_history.history, file_name)       # Save history dict
    plot_Acc_And_Loss(model_history.history)        # Plot acc and loss over epochs

    with open('model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+'_summary', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        cat_dog_model.summary(print_fn=lambda x: fh.write(x + '\n'))
else:
    """Load the model"""
    cat_dog_model = keras.models.load_model('model/'+MODEL_NAME)

    """Load model history and plot loss and acc"""
    with open('model/'+MODEL_NAME+'/trainHistoryDict', 'rb') as file_name:
        model_history = pickle.load(file_name)
    plot_Acc_And_Loss(model_history, save=False)    # Plot but don't save.

    print('\n@> Evaluating model')
    results = cat_dog_model.evaluate(X_val, Y_val, batch_size=32, verbose=1)
    print ("Loss = " + str(results[0]))
    print ("Test Accuracy = " + str(results[1]))

    print('\n@> Predicting Test Data')
    NUM_OF_PRED = 10
    pred = cat_dog_model.predict(X_test_data)

    for i in range(NUM_OF_PRED):
        x = X_test_data[i]
        p = pred[i]
        plt.imshow(x)
        if p[0] > p[1]:
            plt.title("{:.4%} cat | {:.4%} dog -> CAT".format(p[0], p[1]))
        else:
            plt.title("{:.4%} cat | {:.4%} dog -> DOG".format(p[0], p[1]))
        plt.show()