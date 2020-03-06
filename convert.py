import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf 
from numpy import genfromtxt
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import cv2
import os
import csv
def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
if __name__ == "__main__":
 #inports images from same folder
    path = 'POKEMON/'
    files = os.fsencode(path)
    name = []
    for file in os.listdir(files):
        name.append(path + os.fsdecode(file))
        thing = []
    with open('pokemans.csv') as csvfile:
        inp = csv.reader(csvfile, delimiter=',')
        for row in inp:
            if len(row) > 1:
                thing.append(row[1])
            else:
                thing.append(row[0])
    
    #images might not be in order, need to sort them
    name.sort();
    imgList = {}
    num = 0
    #reads in images
    for i in name:
        if '0' in i:
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feature = {'label' : _bytes_feature(thing[num].encode('utf-8')),
                       'image' : _bytes_feature(img.tostring())}
            
    tfrecord_filename = 'pokemans.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

    writer.close()
