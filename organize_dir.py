import os
import shutil

for img_name in os.listdir('datasets/train'):
    if img_name.endswith('.jpg'):
        if 'cat' in img_name:
            shutil.move('datasets/train/'+img_name, 'datasets/train/cat/'+img_name)     # move to cat dir
        elif 'dog' in img_name:
            shutil.move('datasets/train/'+img_name, 'datasets/train/dog/'+img_name)     # move to dog dir