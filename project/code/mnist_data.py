from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from collections import defaultdict

IMG_ROWS, IMG_COLS = 28, 28


class MnistData(object):

    def __init__(self, use_data_subset=False):
        self.n_classes = 10
        self.use_data_subset = use_data_subset

        (x_train, y_train), (x_test, y_test) = self._processed_data(use_data_subset=self.use_data_subset)

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.digit_removed = None

        self.x_digit_train = None
        self.y_digit_train = None

        self.x_digit_test = None
        self.y_digit_test = None

        self.y_train_one_hot = self._one_hot_encode(self.y_train)
        self.y_test_one_hot = self._one_hot_encode(self.y_test)

    def into_fit(self, n=None, generated_data=None):
        if n is not None and self.digit_removed is None:
            raise Exception(
                "must run d.set_removed_class(...) in order to add n samples to data")

        x_train, y_train, x_test, y_test = self._train_test()
        if n is not None:
            n_samples = self.x_digit_train[:n]
            n_labels = self.y_digit_train[:n]

            x_train = np.concatenate((x_train, n_samples))
            y_train = np.concatenate((y_train, n_labels))
            x_train, y_train = self._unison_shuffle(x_train, y_train)

            x_test = np.concatenate((x_test, self.x_digit_test))
            y_test = np.concatenate((y_test, self.y_digit_test))
            x_test, y_test = self._unison_shuffle(x_test, y_test)

        if generated_data is not None:
            if n is None:
                raise Exception(
                    "attempting to train on generated_data with no original samples. nonsense!")
            if x_train[0].shape != generated_data[0].shape:
                print("Training data shape: ", x_train[0].shape)
                print("\nGenerated data shape: ", generated_data[0].shape)
                raise Exception(
                    "The generated_data does not have the same shape as the training data")

            x_train = np.concatenate((x_train, generated_data))
            y_train = np.concatenate((y_train, np.repeat(self.digit_removed, len(generated_data))))
            x_train, y_train = self._unison_shuffle(x_train, y_train)

        return x_train, self._one_hot_encode(y_train), x_test, self._one_hot_encode(y_test)

    def into_evaluate(self):
        if self.digit_removed is not None:
            return self._unison_shuffle(
                np.concatenate((self.x_test, self.x_digit_test)),
                np.concatenate((self.y_test_one_hot, self._one_hot_encode(self.y_digit_test))))
        else:
            return self.x_test, self.y_test_one_hot


    def set_removed_class(self, class_index, verbose=True):
        self._set_removed_digit(digit=class_index)

        train_unique, train_counts = np.unique(
            self.y_train, return_counts=True)
        test_unique, test_counts = np.unique(self.y_test, return_counts=True)

        if verbose:
            print("current number of examples per digit -- train:\n",
                  dict(zip(train_unique, train_counts)))
            print("\ncurrent number of examples per digit -- test:\n",
                  dict(zip(test_unique, test_counts)))

    def get_n_samples(self, n):
        if self.digit_removed is None:
            raise Exception(
                "must run d.set_removed_class(...) in order to get n samples")
        return self.x_digit_train[:n]

    def get_num_classes(self):
        return self.n_classes

    def get_generated_data_stub(self):
        if self.digit_removed is None:
            raise Exception(
                "must run d.set_removed_class(...) in order to get get_generated_data_stub")
        return self.x_digit_train[-50:]

    # this method overrides the disease removed paramater!
    def to_low_shot_dataset(self, verbose=False):
        self.__init__(use_data_subset = False)
        if verbose:
            print("Note class removed paramater was overriden. re-Run d.set_removed_class() if needed")
        
        x, y = self._features_and_labels()

        map_class_to_features = defaultdict(list)

        for _class, features in zip(y, x):
            map_class_to_features[_class].append(features.flatten())

        map_class_to_features = {y: np.array(X)
                                 for y, X in map_class_to_features.items()}

        return map_class_to_features, x[0].shape


#####################################################################
#
#                       PRIVATE METHODS
#
#####################################################################

    def _set_removed_digit(self, digit):
        if self.digit_removed != None:
            self.__init__(use_data_subset=self.use_data_subset)

        if digit is not None:
            self.digit_removed = digit

            where_is_digit = self.y_train[:] == digit
            self.x_digit_train = self.x_train[where_is_digit]
            self.y_digit_train = self.y_train[where_is_digit]

            self.x_train = self.x_train[~where_is_digit]
            self.y_train = self.y_train[~where_is_digit]

            where_is_digit = self.y_test[:] == digit
            self.x_digit_test = self.x_test[where_is_digit]
            self.y_digit_test = self.y_test[where_is_digit]

            self.x_test = self.x_test[~where_is_digit]
            self.y_test = self.y_test[~where_is_digit]

            self.y_train_one_hot = self._one_hot_encode(self.y_train)
            self.y_test_one_hot = self._one_hot_encode(self.y_test)

    def _processed_data(self, use_data_subset):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if use_data_subset:
            x_train, y_train = self._subset(x_train, y_train)
            x_test, y_test = self._subset(x_test, y_test)

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

    def _one_hot_encode(self, y):
        n_rows, n_cols = len(y), self.n_classes
        enc = np.zeros((n_rows, self.n_classes))
        enc[np.arange(n_rows), y] = 1.0
        if self.digit_removed is not None:
            enc = np.delete(enc, [self.digit_removed], axis=1)
        return enc

    def _train_test(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def _unison_shuffle(self, x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def _subset(self, x, y):
        assert len(x) == len(y)
        n = len(x)
        subset_size = round(0.1 * n)
        return x[:subset_size], y[:subset_size]

    def _features_and_labels(self):
        x_train, y_train, x_test, y_test = self._train_test()
        return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
