import os
import cv2
import numpy as np

def list_files(startpath):
	for root, dirs, files in os.walk(startpath):
		return files #Lazy, I know.

def prep_data():
	images = []
	labels = []
	for i in range(1, 37): #Iterate through samples 1 to 37
		directory = "samples/Sample{:03d}".format(i)
		for f in list_files(directory):
			image = cv2.imread(directory+'/'+f)
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			images.append(image_gray)
			labels.append(int_to_label(i))
			cv2.waitKey()
	return images, labels

def int_to_alphanum(num):
	#Integers 0-9 are 1-10
	#Characters A-Z are 11-54
	if num <= 10:
		return "{}".format(num-1)
	else:
		return chr(num+54)

def int_to_label(num):
	label = [0] * 36
	label[num-1] = 1
	return label

def label_to_int(label):
	return np.argmax(label)+1

def label_to_alphanum(label):
	return int_to_alphanum(label_to_int(label))
if __name__ == '__main__':
	print("Preparing Data...")
	images, labels = prep_data()
	print("Data Prepared...")
	for i in range(0, len(images)):
		cv2.imshow("Display", images[i])
		print(label_to_alphanum(labels[i]))
		cv2.waitKey()