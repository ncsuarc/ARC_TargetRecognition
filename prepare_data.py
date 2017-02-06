import os
import cv2
import numpy as np
import tensorflow as tf
import Target

def list_files(startpath):
        for root, dirs, files in os.walk(startpath):
                return files #Lazy, I know.

def prep_data():
        images = []
        labels = []
        f = open("samples/imgpath_label.txt", 'r')
        for line in f:
                filename, label = line[:-1].split(' ')
                image = cv2.imread(filename)
#                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                mask = cv2.inRange(image_gray, 1, 255)
                images.append(image.flatten())
                labels.append(int(label))
        return images, labels

def label_to_alphanumeric(label):
    if type(label) is np.ndarray:
        num = np.argmax(label)
    elif type(label) is int:
        num = label
    else:
        raise TypeError('Only accepts np.ndarray or int')
    name = Target.num_to_alphanumeric(num)
    return name

def label_to_shape(label):
    if type(label) is np.ndarray:
        num = np.argmax(label)
    elif type(label) is int:
        num = label
    else:
        raise TypeError('Only accepts np.ndarray or int')
    name = Target.Shape(num).name
    return name

if __name__ == '__main__':
        print("Preparing Data...")
        images, labels = prep_data()
        print("Data Prepared...")
        step = 0
        batch_size = 36
        while step * batch_size < len(images):
            batch_x = images[step*batch_size:(step+1)*batch_size]
            for i in range(0, len(batch_x)):
                print(label_to_alphanumeric(labels[i]))
                cv2.imshow("Display", images[i].reshape((60,60)))
                cv2.waitKey()
                step += 1
