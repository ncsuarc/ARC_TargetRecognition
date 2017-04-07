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
with open(os.path.join(args['save_location'], 'classes.txt'), 'r') as class_file:
    classes = class_file.read().split('\n')

for subdir, i in zip(classes, range(len(classes))):
    for f in os.listdir(os.path.join(args['input'], subdir)):
        if not (f.endswith('jpg') or f.endswith('png')):
            continue
        img = cv2.imread(os.path.join(args['input'], subdir, f))
        img = cv2.resize(img, (60,60))
        images.append(img.flatten())
        labels.append(i)

images = np.array(images)
labels = np.array(labels)

# Launch the graph
cnn_model = model.Model('training', batch_size=500)
pred_labels = cnn_model.test(images)

correct = 0
total = 0
n = 0
for (img, pred_one_hot, actual) in zip(images, pred_labels, labels):
    predicted = np.argmax(pred_one_hot)
    predicted = classes[predicted] 
    actual = classes[actual] 
    total = total + 1
    if actual == predicted:
        correct = correct + 1
    else:
        if not os.path.isdir("incorrect"):
            os.mkdir("incorrect")
        cv2.imwrite(os.path.join("incorrect", "%s.png") % n, img.reshape((60, 60, 3)))
        n += 1
    print("actual:%s predicted:%s" % (actual, predicted))
    print('Percent Correct:%6.2f%%' % (correct/total * 100))
