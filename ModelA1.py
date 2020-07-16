from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import Model


def modelA1(input_shape):
    X_input = Input(input_shape)

    # Block1
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=1, name='conv2D_0')(X_input)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_0')(X)

    # Block2
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=1, name='conv2D_1')(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(X)

    # Flatten output of ReLU block
    X = Flatten()(X)

    # Fully connected layer with softmax activation
    X = Dense(units=16, activation="relu", name='fc_0')(X)
    X = Dropout(rate=.5)(X)
    X = Dense(units=2, activation="softmax", name='fc_1')(X)

    # Keras model instance
    model = Model(inputs=X_input, outputs=X, name='CatDogModel')
    return model