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

CIFAR10 = 'cifar10'
TRAIN_SIZE = 50000

class Cifar10Data(DataObject):

    def __init__(self, use_data_subset=False, use_features=False, class_removed=None):
        self.train_size = TRAIN_SIZE
        self.name = CIFAR10
        super().__init__(use_data_subset=use_data_subset, use_features = use_features, class_removed = class_removed)

#####################################################################
#
#                       PRIVATE METHODS
#
#####################################################################

    def _load_raw_data(self, class_removed=None):
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

    def get_class_name_by_index(self, class_index):
        return { 0 : 'airplane',
                 1 : 'automobile',
                 2 : 'bird',
                 3 : 'cat',
                 4 : 'deer',
                 5 : 'dog',
                 6 : 'frog',
                 7 : 'horse',
                 8 : 'ship',
                 9 : 'truck',
               }[class_index]
