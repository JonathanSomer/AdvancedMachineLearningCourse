from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from collections import defaultdict
from data_object import *

IMG_ROWS, IMG_COLS = 32, 32
N_CHANNELS = 3

class Cifar10Data(DataObject):

    def __init__(self, use_data_subset=False):
        super().__init__(use_data_subset=use_data_subset)

#####################################################################
#
#                       PRIVATE METHODS
#
#####################################################################

    def _processed_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], N_CHANNELS, IMG_ROWS, IMG_COLS)
            x_test = x_test.reshape(x_test.shape[0], N_CHANNELS, IMG_ROWS, IMG_COLS)
        else:
            x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, N_CHANNELS)
            x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, N_CHANNELS)

        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        # normalize data:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return (x_train, y_train), (x_test, y_test)
