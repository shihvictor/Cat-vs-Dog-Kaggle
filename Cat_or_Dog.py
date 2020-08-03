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
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from model_21 import model_21
from model_22 import model_22
from model_23 import model_23
from model_24 import model_24
from model_25_more_reg import model_25_more_reg


""" CHANGE THESE TO SWITCH BETWEEN TRAINING AND LOADING """
NEW_MODEL = True      # False if loading saved model. True if creating new model.
MODEL_NAME = 'model_25_more_reg_64px'      # model_number_description. If this is changed, make sure to change the compiled model.
PREDICT = False     # False to not predict and save predictions. True to predict and save predictions.
""" HYPERPARAMETERS """
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30

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
    predict_generator = predict_image_datagen.flow_from_directory(directory=TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), class_mode=None, batch_size=32, shuffle=False)
    # Displaying examples in first batch. (also debugging generators)
    batchX = next(predict_generator)  # get first batch of BATCH_SIZE examples.
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX[0].shape, batchX[0].min(), batchX[0].max()))
    for i in range(3):
        img = batchX[0,:,:,:]
        plt.imshow(img)
        # label = batchY[i,:]
        # plt.title('label:'+str(label))
        plt.show()
        batchX = predict_generator.next()
    return train_generator, validation_generator, predict_generator


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
# (BATCH_SIZE, 64, 64, 3) imgs can fit.)
train_generator, validation_generator, predict_generator = create_data_generators()

if NEW_MODEL:
    """Compile Model"""
    cat_dog_model = model_25_more_reg((IMG_SIZE, IMG_SIZE, 3))
    opt = Adam(learning_rate=.0001)  # Set optimizer
    cat_dog_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model
    cat_dog_model.summary()

    """Train the model"""
    model_checkpoint = ModelCheckpoint(filepath='model/'+MODEL_NAME,
                                       verbose=1,
                                       monitor='val_loss',
                                       save_best_only=True)     # Saves lowest val loss full model
    Path('model_logs/'+MODEL_NAME+'_logs/').mkdir(parents=True)     # need to delete this dir if terminate while running.
    csv_logger = CSVLogger(filename='model_logs/'+MODEL_NAME+'_logs/'+MODEL_NAME+'_log.csv', separator=',', append=True)
    # model_history = cat_dog_model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.samples//BATCH_SIZE, epochs=EPOCHS, validation_data=validation_generator, validation_steps=validation_generator.samples//BATCH_SIZE, shuffle=True, callbacks=[csv_logger])
    model_history = cat_dog_model.fit(x=train_generator, epochs=EPOCHS, steps_per_epoch=20000//BATCH_SIZE, callbacks=[csv_logger, model_checkpoint], validation_data=validation_generator, validation_steps=5000//BATCH_SIZE)
    # === save the model ===
    # cat_dog_model.save(filepath='model/' + MODEL_NAME, overwrite=True)

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

    # """Evaluate model"""
    # print('\n@> Evaluating model')
    # # results = cat_dog_model.evaluate_generator(generator=validation_generator, steps=validation_generator.samples, verbose=1)
    # results = cat_dog_model.evaluate(x=validation_dataset, batch_size=32, verbose=1)
    # print ("Loss = " + str(results[0]))
    # print ("Test Accuracy = " + str(results[1]))

    #     """Generate Predictions"""
    print('\n@> Predicting Test Data')
    NUM_OF_PRED = 5
    predictions = cat_dog_model.predict(x=predict_generator, verbose=1)
    formatted_predictions = predictions[:, 1]
    filenames = np.array([int(''.join(filter(str.isdigit, p))) for p in predict_generator.filenames])
    # f_index = filenames.reshape((filenames.shape[0]))
    pred_df_unsorted = pd.DataFrame(data=formatted_predictions, index=filenames, columns=['label'])
    pred_df = pred_df_unsorted.sort_index()
    pred_df.index.name = 'id'
    pred_df.to_csv('prediction_model_20_gpu.csv')

print('DONE')
