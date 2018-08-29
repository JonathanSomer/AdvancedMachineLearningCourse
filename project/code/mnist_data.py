from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from collections import defaultdict
from data_object import *
from data_utils import *

IMG_ROWS, IMG_COLS = 28, 28
TRAIN_SIZE = 60000
MNIST_FEATURES_PICKLE_NAME = 'mnist_features'

class MnistData(DataObject):

    def __init__(self, use_features=False, use_data_subset=False):
        # self.use_features = use_features
        super().__init__(use_data_subset=use_data_subset, use_features=use_features)

#####################################################################
#
#                       PRIVATE METHODS
#
#####################################################################
    # solve this! need to remove 5

    def set_removed_class(self, class_index, verbose=True):
        if self.class_removed != None:
            self.__init__(use_data_subset=self.use_data_subset)

        if class_index is not None:
            self.class_removed = class_index

            class_subset_mask = self.y_train[:] == class_index
            self.x_class_removed_train = self.x_train[class_subset_mask]
            self.y_class_removed_train = self.y_train[class_subset_mask]

            self.x_train = self.x_train[~class_subset_mask]
            self.y_train = self.y_train[~class_subset_mask]

            class_subset_mask = self.y_test[:] == class_index
            self.x_class_removed_test = self.x_test[class_subset_mask]
            self.y_class_removed_test = self.y_test[class_subset_mask]

            self.x_test = self.x_test[~class_subset_mask]
            self.y_test = self.y_test[~class_subset_mask]

    def _processed_data(self):
        if self.use_features is True:
            return self._load_features()
        else:
            return self._load_raw_data()

    def _load_features(self):
        from sklearn.externals import joblib
        dict = joblib.load(read_pickle_path(MNIST_FEATURES_PICKLE_NAME))
        features = dict['features']
        labels = dict['labels']

        x_train = features[:TRAIN_SIZE]
        y_train = labels[:TRAIN_SIZE]

        x_test = features[TRAIN_SIZE:]
        y_test = labels[TRAIN_SIZE:]

        return (x_train, y_train), (x_test, y_test)


    def _load_raw_data(self):    
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(
                x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
            x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
        else:
            x_train = x_train.reshape(
                x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
            x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

        # normalize data:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return (x_train, y_train), (x_test, y_test)
