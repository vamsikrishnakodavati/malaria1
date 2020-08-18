from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix
from keras.utils.data_utils import Sequence
#import present_confusion_matrix
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


import matplotlib.pyplot as plt
from itertools import cycle

np.random.seed(8468)

im_ht = im_wid = 210
batch_size = 16
num_classes = 2



class DataGenerator(Sequence):
    def __init__(self, data_set, batch_size):
        self.Data_Set = data_set
        print("Class Distribution:", [len(_[0]) for _ in self.Data_Set])
        print("Max data for a class:", max([len(_[0]) for _ in self.Data_Set]))
        self.batch_size = batch_size

    def __len__(self):
        return int(max([len(_[0]) for _ in self.Data_Set]) // self.batch_size)

    def __getitem__(self, index):

        X = []
        Y = []
        for _class_index in range(len(self.Data_Set)):
            index_ = int(index % (int(len(self.Data_Set[_class_index][0]) / self.batch_size)))

            if index_ == 0:
                suffle_index = np.arange(len(self.Data_Set[_class_index][0]))
                np.random.shuffle(suffle_index)
     \           self.Data_Set[_class_index][0] = np.array(self.Data_Set[_class_index][0])[suffle_index]
                self.Data_Set[_class_index][1] = np.array(self.Data_Set[_class_index][1])[suffle_index]

            x, y = self.Data_Set[_class_index]

            batch_x = x[index_ * self.batch_size: (index_ + 1) * self.batch_size]
            batch_y = y[index_ * self.batch_size: (index_ + 1) * self.batch_size]

            for i in range(len(batch_x)):
                # ------------------------------------------------------------------------------------------------------------------------------------
                im_good = batch_x[i]
                im_good = cv2.warpAffine(im_good,
                                         cv2.getRotationMatrix2D((im_good.shape[1] / 2, im_good.shape[0] / 2), np.random.choice([0, 1, -1, 2]) * 90,
                                                                 1),
                                         (im_good.shape[1], im_good.shape[0]))
                im_good = cv2.flip(im_good, np.random.choice([0, 1, -1, 2]))
                X.append(im_good)
                Y.append(batch_y[i])


        suffle_index = np.arange(len(X))
        np.random.shuffle(suffle_index)
        X = np.array(X)[suffle_index]
        Y = np.array(Y)[suffle_index]

        return [X, Y]


def one_hot(pos):
    a = np.zeros(num_classes)
    a[pos] = 1
    return a


print("Loading Data!")
x = []
y = []
all_filenames = []

load_data = "./cell_images/"

data = []

for classes in os.listdir(load_data):
    x = []
    y = []
    for image in os.listdir(load_data + "/" + classes):
        a = load_data + "/" + classes + "/" + image
        b = a.split(".")[-1]
        if b == 'png':
         im = cv2.imread( a )
         all_filenames.append(a)


        # im = cv2.imread(load_data + "/" + classes + "/" + image, 0)
         print (image)

        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
         im = cv2.resize(im, (210, 210))
         im = im/255
         im = (im - np.min(im)) / (np.max(im) - np.min(im))
         x.append(im)
         y.append(one_hot(int(classes)))
    data.append([x, y])

data = np.array(data)

all_filenames = np.array(all_filenames)

train_data = np.array([[x[:int(len(x) * 0.7)], y[:int(len(y) * 0.7)]] for x, y in data])
train_gen = DataGenerator(train_data, batch_size)
train_filenames = all_filenames[:int(len(x) * 0.7)]

valid_data_temp = np.array([[x[int(len(x) * 0.7):int(len(x) * 0.9)], y[int(len(y) * 0.7):int(len(y) * 0.9)]] for x, y in data])

valid_data = [[], []]
for x, y in valid_data_temp:
    [valid_data[0].append(_) for _ in x]
    [valid_data[1].append(_) for _ in y]

valid_data[0] = np.array(valid_data[0])
valid_data[1] = np.array(valid_data[1])


test_data_temp = np.array([[x[int(len(x) * 0.9):int(len(x) * 1)], y[int(len(y) * 0.9):int(len(y) * 1)]] for x, y in data])

test_data = [[], []]
for x, y in test_data_temp:
    [test_data[0].append(_) for _ in x]
    [test_data[1].append(_) for _ in y]

test_data[0] = np.array(test_data[0])
test_data[1] = np.array(test_data[1])


valid_filenames = all_filenames[int(len(x) * 0.7):int(len(x) * 1)]

print(train_gen[0][0].shape, train_gen[0][1].shape)

num_batches_train = len(train_gen)
num_batches_valid = len(valid_data[0]) // batch_size + 1

class_sum = np.zeros(num_classes)
for i in range(num_batches_train):
    class_sum += np.sum(train_gen[i][1], axis=0)
    # [print(train_gen[i][0][_], train_gen[i][1][_]) for _ in range(len(train_gen[i][0]))]

print("Distribution accross all classes (Generated Data):", class_sum)

print("Dataset Size:", train_data.shape, len(valid_data[0]))
print("Number of batches:", num_batches_train, num_batches_valid)
print("Valid Shape:", valid_data[0].shape, valid_data[1].shape, test_data[0].shape, test_data[1].shape)



def get_model(parameters):
    base_model = InceptionV3(input_shape=(im_ht, im_wid, 3), weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for _ in parameters["nodes_layer"]:
        x = Dense(_)(x)
        x = Dropout(parameters["dropout_prob"])(x)
        x = Activation('relu')(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # parameters["trainable_layers"] = 311 - parameters["trainable_layers"]
    for layer in base_model.layers[:parameters["trainable_layers"]]:
        layer.trainable = False
    for layer in base_model.layers[parameters["trainable_layers"]:]:
        layer.trainable = True

    optim = SGD(parameters["learning_rate"], 0.9, nesterov=True, clipvalue=100.)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model_param = {'learning_rate': 0.005, 'dropout_prob': 0.6, 'nodes_layer': [], 'trainable_layers': 0}

model = get_model(model_param)

print(model.summary())
# MODEL END =====================================================================================
# Training =====================================================================================

class Confusion_Mat_Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 5 == 0:
            preds = model.predict(valid_data[0])
            preds = np.argmax(preds, axis=-1)
            temp = np.argmax(valid_data[1], axis=-1)
            print("=" * 50, "\n", confusion_matrix(temp, preds))
        return



checkpoint = callbacks.ModelCheckpoint("./tmp/modelmal5.hdf5", monitor='val_loss', verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False, mode='auto', period=1)
checkpoint2 = callbacks.ModelCheckpoint("./tmp/modelmal5.hdf5", monitor='val_acc', verbose=0,
                                        save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1)

tfboard = callbacks.TensorBoard(log_dir='./tmp/logsa2' + datetime.datetime.now().strftime("%H-%M-%S"), histogram_freq=0, batch_size=30,
                                write_graph=True, write_grads=True, write_images=True, embeddings_freq=0,
                                embeddings_layer_names=None, embeddings_metadata=None)




conf_call = Confusion_Mat_Callback()

callback = [checkpoint, checkpoint2, tfboard, conf_call]

print("Starting to train!")
# model.fit_generator(train_gen, steps_per_epoch=num_batches_train, epochs=100, validation_data=valid_data, callbacks=callback)

print("Training End =====================================================================================")
print("Testing =====================================================================================")
model.load_weights("./tmp/modelmal5.hdf5")

print("Testing:")

x = test_data[0]
y = test_data[1]
print(x.shape, y.shape)
# print(model.evaluate(x, y, verbose=0))
predictions = model.predict(x)

preds = np.argmax(predictions, axis=-1)

y_real = np.argmax(y, axis=-1)

print(y_real)
print(preds)
print(len(preds), len(y_real))


conf_mat = confusion_matrix(y_true=y_real, y_pred=preds)

present_confusion_matrix.CMnLoss(conf_mat)
present_confusion_matrix.CMnLoss(conf_mat, False)

