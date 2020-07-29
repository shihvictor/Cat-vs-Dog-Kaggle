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
from keras.preprocessing.image import ImageDataGenerator


""" CHANGE THESE TO SWITCH BETWEEN TRAINING AND LOADING """
NEW_MODEL = True      # False if loading saved model. True if creating new model.
MODEL_NAME = 'model_9_testrun2'

""" HYPERPARAMETERS """
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

""" FILE ..."""
TRAIN_DIR = 'datasets/train'
VALIDATION_DIR = 'datasets/validation'
TEST_DIR = 'datasets/test'
# print(os.listdir(TRAIN_DIR))

""" CHANGE THIS TO USE VALIDATION DATASET """
def create_data_generators():
    """
    Generates training, validation, and prediction generators to load entire datasets without exceeding RAM.
    :return: A tuple (train_generator, validation_generator, predict_generator), where the train and validation
    generators yield batches of image data and labels and predict generator yields images.
    """
    train_image_datagen = ImageDataGenerator(rescale=1./255)
    validation_image_datagen = ImageDataGenerator(rescale=1./255)
    predict_image_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_image_datagen.flow_from_directory(directory=TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), class_mode='categorical', batch_size=BATCH_SIZE, shuffle=True)
    validation_generator = validation_image_datagen.flow_from_directory(directory=VALIDATION_DIR, target_size=(IMG_SIZE, IMG_SIZE), class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False)
    predict_generator = predict_image_datagen.flow_from_directory(directory=TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), class_mode=None, batch_size=1, shuffle=False)
    # debugging generators
    # batchX, batchY = train_generator.next()
    batchX, batchY = next(train_generator)
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX[0].shape, batchX[0].min(), batchX[0].max()))
    for i in range(5):
        img = batchX[i,:,:,:]
        plt.imshow(img)
        label = batchY[i,:]
        plt.title('label:'+str(label))
        plt.show()
        batchX, batchY = train_generator.next()
    return train_generator, validation_generator, predict_generator


def label_img(img_name):
    """ Deprecated.
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
    """ Deprecated.
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
    test_imgs = os.listdir(TEST_DIR)
    test_imgs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for img_name in tqdm(test_imgs):
        # read and resize img for ex i = x(i)
        image = np.array(plt.imread(TEST_DIR + '/' + img_name))
        x = resize(image, [IMG_SIZE, IMG_SIZE, 3])  # resizes the dim of the image array to fit to CNN

        # plt.imshow(x, interpolation='nearest')
        # plt.show()
        # Save features and label
        X_test.append(x)

        # plt.imshow(x)
        # plt.show()
    # Uncomment this later to randomize order of imgs.
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

    # Block2
    X = Conv2D(filters=64, kernel_size=(5, 5), strides=1, name='conv2D_1')(X)
    X = BatchNormalization(axis=3, name='bn_1')(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(X)

    # Block3
    X = Conv2D(filters=128, kernel_size=(5, 5), strides=1, name='conv2D_2')(X)
    X = BatchNormalization(axis=3, name='bn_2')(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(X)

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


# Generates batches from datasets. (Saving the entire train set as (20k, 128, 128, 3) is too large to store in memory.
# (64, 64, 3) imgs can fit.)
train_generator, validation_generator, predict_generator = create_data_generators()

if NEW_MODEL:
    """Compile Model"""
    cat_dog_model = model((IMG_SIZE, IMG_SIZE, 3))
    opt = Adam(learning_rate=.0001)  # Set optimizer
    cat_dog_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model
    cat_dog_model.summary()

    """Train the model"""
    Path('model_logs/'+MODEL_NAME+'_logs/').mkdir(parents=True)
    csv_logger = CSVLogger(filename='model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+'_log.csv', separator=',', append=True)
    # model_history = cat_dog_model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.samples//BATCH_SIZE, epochs=EPOCHS, validation_data=validation_generator, validation_steps=validation_generator.samples//BATCH_SIZE, shuffle=True, callbacks=[csv_logger])
    model_history = cat_dog_model.fit(x=train_generator, epochs=EPOCHS, steps_per_epoch=20000//BATCH_SIZE, callbacks=[csv_logger], validation_data=validation_generator, validation_steps=5000//BATCH_SIZE)
    # === save the model ===
    cat_dog_model.save(filepath='model/' + MODEL_NAME, overwrite=True)

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
    # Load model history and plot loss and acc
    with open('model/'+MODEL_NAME+'/trainHistoryDict', 'rb') as file_name:
        model_history = pickle.load(file_name)
    plot_Acc_And_Loss(model_history, save=False)    # Plot but don't save.

    """Evaluate model"""
    print('\n@> Evaluating model')
    # results = cat_dog_model.evaluate_generator(generator=validation_generator, steps=validation_generator.samples, verbose=1)
    results = cat_dog_model.evaluate(x=validation_dataset, batch_size=32, verbose=1)
    print ("Loss = " + str(results[0]))
    print ("Test Accuracy = " + str(results[1]))
#
#     """Generate Predictions"""
#     print('\n@> Predicting Test Data')
#     NUM_OF_PRED = 5
#     predict_generator.reset()
#     t11 = predict_generator.samples
#     pred = cat_dog_model.predict_generator(generator=predict_generator, steps=predict_generator.samples, verbose=1)

    # filenames = [int(''.join(filter(str.isdigit, p))) for p in predict_generator.filenames]
    # predictions = pd.DataFrame({"id": filenames, "label": pred[:, 1]})
    # predictions.sort_values(by="id", inplace=True)
    # predictions.reset_index(inplace=True)
    # print(predictions.head())
    # predictions.to_csv('unsorted_results.csv')

print('DONE')
