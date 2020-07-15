from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import Model


def model12(input_shape):
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