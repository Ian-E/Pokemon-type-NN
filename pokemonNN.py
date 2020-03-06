import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow.contrib.slim as slim
import numpy as np

from numpy import genfromtxt
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import cv2
import os
import csv
#/============================================================/#
if __name__ == "__main__":

    model = Sequential()
    #inports images from same folder
    path = 'POKEMON/'
    files = os.fsencode(path)
    name = []
    for file in os.listdir(files):
        name.append(path + os.fsdecode(file))
    
    #images might not be in order, need to sort them
    name.sort();
    imgList = []
    #reads in images
    for i in name:
        if '0' in i:
             imgList.append(cv2.imread(i))
    x = np.array(imgList)
    print(x)
    #reads in types csv's
    thing = []
    with open('pokemans.csv') as csvfile:
        inp = csv.reader(csvfile, delimiter=',')
        for row in inp:
            if len(row) > 1:
                thing.append(row[1])
            else:
                thing.append(row[0])
    y = np.array(thing)
    with open('labels.txt', 'w') as f:
        for item in thing:
            f.write("%s\n" % item)
    input_shape = (215,215,3)
    num_classes = 18


# 64 channels, filter window is 5x5, strides are 1x1.
# Input shape is the size of the image.
#    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))

# Use max pooling window of 2x2, strides are 2x2.
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 32 channels, kernal size still 5x5.
#    model.add(Conv2D(32, (5, 5), activation='relu'))

# Same as before?
#    model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten from 3d to 1d
#    model.add(Flatten())


#    model.add(Dense(80, activation='relu'))
#    model.add(Dense(num_classes, activation='softmax'))

#    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


    #model.fit(train, labels, epochs=10, batch_size=20)

