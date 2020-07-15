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
import tensorflow as tf
from tensorflow import keras
import pickle

TRAIN_DIR = 'datasets/train'
TEST_DIR = 'datasets/test'
IMG_SIZE = 64
print(os.listdir(TRAIN_DIR))


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


# def create_train_data():
#     training_data = []
#     i = 0
#     for img_name in tqdm(os.listdir(TRAIN_DIR)):
#         label = np.array([label_img(img_name)])     # [] makes shape (1, 2) instead of (2,)
#         image = np.array(plt.imread(TRAIN_DIR+'/'+img_name))
#         resized_image = resize(image, [IMG_SIZE, IMG_SIZE, 3]) # resizes the dim of the image array to fit to CNN
#         plt.imshow(resized_image, interpolation='nearest')
#         plt.show()
#         training_data.append([resized_image, label])
#         i += 1
#         if i == 5: break
#     # shuffle(training_data)
#     training_data = np.array(training_data)
#     np.save('datasets/train_data.npy', training_data)
#     return training_data


def process_test_data():
    X_test = []
    i = 0
    for img_name in tqdm(os.listdir(TEST_DIR)):
        # read and resize img for ex i = x(i)
        image = np.array(plt.imread(TEST_DIR + '/' + img_name))
        x = resize(image, [IMG_SIZE, IMG_SIZE, 3])  # resizes the dim of the image array to fit to CNN

        # plt.imshow(x, interpolation='nearest')
        # plt.show()
        # Save features and label
        X_test.append(x)
        i += 1
        if i == 5: break
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


# def partition_data(X_data, Y_data, part_amounts):
#     """
#
#     :param X_data: Numpy array. All X data that the model will use
#     :param Y_data: Numpy array. All X data that the model will use
#     :param part_amounts: tuple
#     :return: tuple of the partitions
#     """
#     M = X_data.shape[0]
#
#     if len([part_amounts]) == 1:
#         return (X_data[0:int(M*.8), :])


def plot_Acc_And_Loss(history_dict):
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
    plt.show()

    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


""" LOAD DATA """
# === Either have (1) or (2) commented.
# X_data, Y_data = create_train_data()  # (1)
# === If train data .npy file already created, load it instead.
X_data = np.load('datasets/X_train_data.npy')   #(2)
Y_data = np.load('datasets/Y_train_data.npy')   #(2)

M = X_data.shape[0]
X_train = X_data[0:int(M*.8), :]    # Train and cv data
Y_train = Y_data[0:int(M*.8), :]
X_test = X_data[int(M*.8):, :]
Y_test = Y_data[int(M*.8):, :]
print("number of training examples : " + str(X_train.shape[0]))
print("number of test examples : " + str(X_test.shape[0]))
print("X_train shape : " + str(X_train.shape))
print("Y_train shape : " + str(Y_train.shape))
print("X_test shape : " + str(X_test.shape))
print("Y_test shape : " + str(Y_test.shape))

"""CHANGE THESE TO SWITCH BETWEEN TRAINING AND LOADING"""
NEW_MODEL = True
MODEL_NAME = 'my_model_11'


# Compile, Train, Save, Plot
if NEW_MODEL:
    """Compile Model"""
    cat_dog_model = model(X_train.shape[1:])
    cat_dog_model.summary()
    cat_dog_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    """Train the model"""
    # may have to use own CV set instead of using validation_split.
    model_history = cat_dog_model.fit(x=X_train, y=Y_train, batch_size=32, epochs=15, validation_split=.2, shuffle=True)  # val split should be .25
    #=== save the model ===
    cat_dog_model.save(filepath='model/'+MODEL_NAME)
#=== Or load the model ===
# cat_dog_model = keras.models.load_model('model/'+MODEL_NAME)


    """Save model history and plot loss and acc"""
    with open('model/'+MODEL_NAME+'/trainHistoryDict', 'wb') as file_name: # opens file from /test1 and saves to file_pi
        pickle.dump(model_history.history, file_name) # saves history to path file_pi
    plot_Acc_And_Loss(model_history.history)



"""Load model history and plot lost and acc"""
# with open('model/'+MODEL_NAME+'/trainHistoryDict', 'rb') as file_name:
#     model_history = pickle.load(file_name)
#
# # print(model_history.keys())
# plt.plot(model_history['accuracy'])
# plt.plot(model_history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.plot(model_history['loss'])
# plt.plot(model_history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


# print('\n@> Evaluating model')
# results = cat_dog_model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
# print ("Loss = " + str(results[0]))
# print ("Test Accuracy = " + str(results[1]))
#
#
# print('\n@> Predicting Test Data')
# NUM_OF_PRED = 10
# pred = cat_dog_model.predict(X_test[:NUM_OF_PRED])
#
# for i in range(NUM_OF_PRED):
#     x = X_test[i]
#     p = pred[i]
#     plt.imshow(x)
#     if p[0] > p[1]:
#         plt.title("{:.4%} cat | {:.4%} dog -> CAT".format(p[0], p[1]))
#     else:
#         plt.title("{:.4%} cat | {:.4%} dog -> DOG".format(p[0], p[1]))
#
#     plt.show()


