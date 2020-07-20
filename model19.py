from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import Model


def model19(input_shape):
    """
    Keras model
    :return: Keras model using Keras functional API https://keras.io/guides/functional_api/
    """
    # Input layer
    X_input = Input(input_shape)

    # Block0
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(X_input)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    # Block1
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    # Block2
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    # Block3
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(rate=.2)(X)

    # Flatten output of ReLU block
    X = Flatten()(X)

    # Fully connected layer with softmax activation
    X = Dense(units=256, activation="relu")(X)
    X = Dropout(rate=.5)(X)
    X = Dense(units=256, activation="relu")(X)
    X = Dropout(rate=.5)(X)
    X = Dense(units=2, activation="softmax")(X)


    # Keras model instance
    model = Model(inputs=X_input, outputs=X, name='CatDogModel_19')
    return model