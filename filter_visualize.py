from tensorflow import keras
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input
from numpy import expand_dims
from skimage.transform import resize
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = 'model_17'

img0 = np.array(plt.imread('datasets/train/cat.1.jpg'))
img0 = resize(img0, [64, 64, 3])
plt.imshow(img0)
# plt.show()

img = load_img('datasets/train/cat.1.jpg', target_size=(64, 64))
img = img_to_array(img) / 255.
plt.imshow(img)
plt.show()

img = expand_dims(img, axis=0)

"""Load the model"""
cat_dog_model = keras.models.load_model('model/' + MODEL_NAME)

"""Create new model with output of the original model's conv2d_0"""
# idxs = [2, 8, 14, 20]
# outputs = [cat_dog_model.layers[i].output for i in idxs]
cat_dog_model = Model(inputs=cat_dog_model.input, outputs=cat_dog_model.layers[20].output)

"""Get feature map for conv2d_0"""
feature_maps = cat_dog_model.predict(x=img)

square = 8
for fmap in feature_maps:
    ix = 1
    for _ in range(4):
        for _ in range(8):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    print("")
