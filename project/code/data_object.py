from abc import ABCMeta, abstractmethod
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from collections import defaultdict


class DataObject(object):
    def __init__(self, use_data_subset=False):
        self.use_data_subset = use_data_subset
        (x_train, y_train), (x_test, y_test) = self._processed_data(use_data_subset=self.use_data_subset)

        self.n_classes = len(np.unique(y_test))
        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.class_removed = None
        self.x_class_removed_train = None
        self.y_class_removed_train = None
        
        self.x_class_removed_test = None
        self.y_class_removed_test = None

        self.y_train_one_hot = self._one_hot_encode(self.y_train)
        self.y_test_one_hot = self._one_hot_encode(self.y_test)

        self.number_of_samples_to_use = None
        self.generated_data = None

    @abstractmethod
    def _processed_data(self, use_data_subset):
        raise NotImplementedError("abstract class")

    @abstractmethod
    def _one_hot_encode(self, y):
        raise NotImplementedError("abstract class")     

