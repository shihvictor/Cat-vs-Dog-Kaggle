import os
import shutil
from random import shuffle
import numpy as np

n = 0
if n == 1:
    for img_name in os.listdir('datasets/train'):
        if img_name.endswith('.jpg'):
            if 'cat' in img_name:
                shutil.move('datasets/train/'+img_name, 'datasets/train/cat/'+img_name)     # move to cat dir
            elif 'dog' in img_name:
                shutil.move('datasets/train/'+img_name, 'datasets/train/dog/'+img_name)     # move to dog dir
elif n == 2:
    for img_name in os.listdir('datasets/train/cat'):
        if img_name.endswith('.jpg'):
            shutil.move('datasets/train/cat/'+img_name, 'datasets/train/'+img_name)
    for img_name in os.listdir('datasets/train/dog'):
        if img_name.endswith('.jpg'):
            shutil.move('datasets/train/dog/'+img_name, 'datasets/train/'+img_name)

def partition_train_data():
    cat_names = [img_name for img_name in os.listdir('datasets/train/cat') if 'jpg' in img_name]
    shuffle(cat_names)
    i = 0 # Number of moved images
    for cat_name in cat_names:
        shutil.move('datasets/train/cat/'+cat_name, 'datasets/validation/cat/'+cat_name)
        i += 1
        if i == 2500: break

    dog_names = [img_name for img_name in os.listdir('datasets/train/dog') if 'jpg' in img_name]
    shuffle(dog_names)
    i = 0 # Number of moved images
    for dog_name in dog_names:
        shutil.move('datasets/train/dog/'+dog_name, 'datasets/validation/dog/'+dog_name)
        i += 1
        if i == 2500: break

    print("DONE PARTITIONING")

partition_train_data()