#!/usr/bin/env python
# -- coding: UTF-8 --

"""
Usage:
    main.py -r <img_rows> -c <img_columns> [-b <batch_normalization>] [-e <epochs>] [-s <val_split>] <activation> <optimizer> <initializer>

Options:
  -r --rows <img_rows>              Set the number of rows in the pixel array of each image.
  -c --cols <img_cols>              Set the number of columns in the pixel array of each image.
  -b --batchn=<0>                   Decide whether to apply batch normalization between layers.[default: 0]
  -e --epochs=<20>                  Set the desired level of epochs.[default: 20]
  -s --valsplit=<0.1>               Set the size of the validation set as a percentage to the input. [default: 0.1]
  -v --version                      Show program's version number and exit
  -h --help                         Show this help message and exit

Available <activation> functions as per keras:
  softmax
  elu
  selu
  softplus
  softsign
  relu
  tanh
  sigmoid
  hard_sigmoid
  exponential
  linear

Available <optimizer> functions as per keras:
  sgd
  rmsprop
  adagrad
  adadelta
  adam
  adamax
  nadam

Available <initializer> functions as per keras:
  random_uniform
  glorot_uniform
  glorot_normal
  he_normal
  lecun_normal
  he_uniform
"""

from __future__ import print_function
from docopt import docopt
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from data_handler import Data, plot_acc, plot_tt_acc
from keras.regularizers import l2
import matplotlib.pyplot as plt


##################################
# Get command line arguments.
##################################
args            = docopt(__doc__, version= 'emotionCapture 0.0.1')
#input_file      = args["--input"]
img_rows        = int(args["--rows"])
img_cols        = int(args["--cols"])
epochs          = int(args["--epochs"])
batch_norm      = True if int(args["--batchn"]) == 1 else False
val_split       = float(args["--valsplit"])
optimizer       = args["<optimizer>"]
activation_func = args["<activation>"]
initializer     = args["<initializer>"]
##################################

batch_size = 128
num_classes = 11
#epochs = 20

# input image dimensions
# img_rows, img_cols = 50, 50

output_file_name = optimizer + "_" + activation_func + "_" + initializer + "_" + args["--batchn"] + "_" + str(epochs) + ".h5"

print("Results will be saved in models/" + output_file_name)

print("Will create a CNN the following options:")
print(args)

data = Data(7000, False)

x_train, y_train = data.sample_train()
x_test, y_test = data.sample_test()
class_names = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'No-Face']


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

#1
model.add(Conv2D(32, kernel_size=(3, 3),
                 input_shape=input_shape,
                 kernel_initializer=initializer,
                 kernel_regularizer=l2(0.01)))

if batch_norm:
  model.add(BatchNormalization())
model.add(Activation(activation_func))
model.add(Dropout(0.25))


#2
model.add(Conv2D(64, 
                 kernel_size=(3, 3),
                 input_shape=input_shape,
                 kernel_initializer=initializer,
                 kernel_regularizer=l2(0.01)))

if batch_norm:
  model.add(BatchNormalization())
model.add(Activation(activation_func))
model.add(Dropout(0.25))

#3
model.add(Conv2D(128,
                 kernel_size=(3, 3),
                 kernel_initializer=initializer,
                 kernel_regularizer=l2(0.01)))

if batch_norm:
  model.add(BatchNormalization())
model.add(Activation(activation_func))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, kernel_initializer=initializer,
                     kernel_regularizer=l2(0.01)))

if batch_norm:
  model.add(BatchNormalization())

model.add(Activation(activation_func))
model.add(Dropout(0.25))

model.add(Dense(128, kernel_initializer=initializer,
                     kernel_regularizer=l2(0.01)))

if batch_norm:
  model.add(BatchNormalization())

model.add(Activation(activation_func))
model.add(Dropout(0.25))

model.add(Dense(num_classes, 
                     kernel_initializer=initializer,
                     kernel_regularizer=l2(0.01)))

if batch_norm:
    model.add(BatchNormalization())
model.add(Activation('softmax'))

if optimizer == "sgd":
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
elif optimizer == "rmsprop":
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
elif optimizer == "adagrad":
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(),
              metrics=['accuracy'])
elif optimizer == "adadelta":
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
elif optimizer == "adam":
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
elif optimizer == "nadam":
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split= 0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#model.save('/Users/Chiara/Desktop/Emotion_recognition_CNN/models/Tanh_WeightNormal_Batch_L21')

model.save('models/' + output_file_name)
