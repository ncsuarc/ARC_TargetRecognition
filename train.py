import cv2
import tensorflow as tf
import numpy as np

import model

import argparse
import os

parser = argparse.ArgumentParser(description='Load and train a model.')
parser.add_argument("-i", "--input", required=True, help="Directory to find training images in.")
parser.add_argument("-s", "--save-location", required=True, help="Directory to save and load model in.")
args = vars(parser.parse_args())

images = []
labels = []

subdirs = [name for name in os.listdir(args['input']) if os.path.isdir(os.path.join(args['input'], name))]
for subdir, i in zip(subdirs, range(len(subdirs))):
    for f in os.listdir(os.path.join(args['input'], subdir)):
        if not (f.endswith('jpg') or f.endswith('png')):
            continue
        img = cv2.imread(os.path.join(args['input'], subdir, f))
        img = cv2.resize(img, (60,60))
        images.append(img.flatten())
        labels.append(i)

images = np.array(images)
labels = np.array(labels)

rng_state = np.random.get_state()
np.random.shuffle(images)
np.random.set_state(rng_state)
np.random.shuffle(labels)

with open(os.path.join(args['save_location'], 'classes.txt'), 'w+') as class_file:
    class_file.write('\n'.join(subdirs))

# Launch the graph
cnn_model = model.Model(args['save_location'], batch_size = 500, display_step = 10)
cnn_model.train(images, labels, len(subdirs))
