from __future__ import print_function
import math
from flask import Flask, request
import cv2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam, SGD
from keras.utils.data_utils import Sequence
from keras.models import Model
from keras import callbacks
import keras.backend as K
import numpy as np
import datetime
import keras
import time
import math
import cv2
import os
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.keras.models import load_model


import pickle
import os
import json
import pandas as pd
import nltk
import numpy as np
import random
import string
import operator
from flask import render_template

def get_img_predict(image1):
    np.random.seed(8468)
    im_ht = im_wid = 210  #initialization image size
    batch_size = 16
    num_classes = 2 #initialization number of classes


    def one_hot(pos):    # one hot encoding (category is represented as a one-hot vector)
        a = np.zeros(num_classes)
        a[pos] = 1
        return a

    print("Loading Data!")  #initialization x- image y-lables
    x = []
    y = []
    all_filenames = []

    load_data = "./mal1/"

    data = []

    for classes in os.listdir(load_data): #read from folder mal1 folder
        x = []
        y = []
        for q in os.listdir(load_data + "/" + classes):
            a = load_data + "/" + classes + "/" + q
            b = a.split(".")[-1]
            if b == 'png':
                im = cv2.imread(image1) # app.py uplode folder
                all_filenames.append(a)

                # im = cv2.imread(load_data + "/" + classes + "/" + image, 0)


                # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                im = cv2.resize(im, (210, 210)) #resize
                im = im / 255
                im = (im - np.min(im)) / (np.max(im) - np.min(im)) #normalize function image1
                x.append(im)
                y.append(one_hot(int(classes) - 1))
        data.append([x, y])

    data = np.array(data)

    test_data_temp = np.array(
        [[x[int(len(x) * 0):int(len(x) * 1)], y[int(len(y) * 0):int(len(y) * 1)]] for x, y in data])

    test_data = [[], []]  # test data appending  x,y
    for x, y in test_data_temp:
        [test_data[0].append(_) for _ in x]
        [test_data[1].append(_) for _ in y]

    test_data[0] = np.array(test_data[0]) # numpy array conversition
    test_data[1] = np.array(test_data[1])
    print("Starting to train!")
    # model.fit_generator(train_gen, steps_per_epoch=num_batches_train, epochs=100, validation_data=valid_data, callbacks=callback)
    base_model = InceptionV3(input_shape=(im_ht, im_wid, 3), weights='imagenet', include_top=False) #inception achitecture loading
    x = base_model.output
    x = GlobalAveragePooling2D()(x)



    predictions = Dense(num_classes, activation='softmax')(x)  #using softmax activation to get image preictions

    model = Model(inputs=base_model.input, outputs=predictions)



    optim = SGD(0.001, 0.9, nesterov=True, clipvalue=100.) # load parameters used in training
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) # Loss  is calculated based on the uncertainty of how much the prediction changes from the true value





    model_param = {'learning_rate': 0.005, 'dropout_prob': 0.6, 'nodes_layer': [], 'trainable_layers': 0} # intializing the learning rate


    # print(model.summary())

    # model_param = {'learning_rate': 0.005, 'dropout_prob': 0.6, 'nodes_layer': [], 'trainable_layers': 0}


    print("Training End =====================================================================================")
    print("Testing =====================================================================================")
    model.load_weights("./tmp/modelmal5.hdf5")  # loading the weights which are trained

    print("Testing:")

    x = test_data[0]
    y = test_data[1]
    print(x.shape, y.shape)
    # print(model.evaluate(x, y, verbose=0))
    predictions = model.predict(x)

    preds = np.argmax(predictions, axis=-1)  # Max fuction if vale >0.5 assign 1 or else 0
    print(preds[0])

    if (preds[0] == 1):
        result = "Parasitized"
        print(result)
    elif (preds[0]  == 0):
        result = "Uninfected"
        print(result)
    return result



# get_img_predict('./mal1')
