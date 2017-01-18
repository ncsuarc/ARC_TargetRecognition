import os
import cv2
import numpy as np
import tensorflow as tf

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
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask = cv2.inRange(image_gray, 1, 255)
                images.append(mask.flatten())
                labels.append(int_to_label(int(label)))
        return images, labels

def int_to_shape(num):
    return {
            1:'circle',
            2:'semicircle',
            3:'quarter_circle',
            4:'triangle',
            5:'square',
            6:'rectangle',
            7:'trapezoid',
            8:'pentagon',
            9:'hexagon',
            10:'heptagon',
            11:'octagon',
            12:'star',
            13:'cross'
    }.get(num, -1)

def int_to_label(num):
        label = [0] * 13
        label[num] = 1
        return label

def label_to_int(label):
        return np.argmax(label)

def label_to_shape(label):
        return int_to_shape(label_to_int(label))
        
if __name__ == '__main__':
        print("Preparing Data...")
        images, labels = prep_data()
        print("Data Prepared...")
        step = 0
        batch_size = 36
        while step * batch_size < len(images):
            batch_x = images[step*batch_size:(step+1)*batch_size]
            for i in range(0, len(batch_x)):
                print(label_to_shape(labels[i]))
                cv2.imshow("Display", images[i].reshape((60,60)))
                cv2.waitKey()
                step += 1
