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
from data_utils import *
from sklearn.externals import joblib
import random

FRACTION_FOR_SUBSET = 0.05

class DataObject(object):
    def __init__(self, use_data_subset=False, use_features=False, class_removed=None):
        self.use_data_subset = use_data_subset
        self.use_features = use_features

        (x_train, y_train), (x_test, y_test) = self._processed_data(class_removed=class_removed)

        assert len(np.unique(y_test)) == len(np.unique(y_train))
        self.n_classes = len(np.unique(y_test))

        if self.use_data_subset:
            x_train, y_train = self._subset(x_train, y_train)
            x_test, y_test = self._subset(x_test, y_test)

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test[:-1000]
        self.y_test = y_test[:-1000]

        self.x_dev = x_test[-1000:]
        self.y_dev = y_test[-1000:]


        self.class_removed = None
        self.x_class_removed_train = None
        self.y_class_removed_train = None
        
        self.x_class_removed_test = None
        self.y_class_removed_test = None


        self.y_train_one_hot = self._one_hot_encode(self.y_train)
        self.y_test_one_hot = self._one_hot_encode(self.y_test)

        self.number_of_samples_to_use = None
        self.generated_data = None
        
        self.permutation_for_sample_fetch = np.random.permutation(3000)

    def into_fit(self, fix_class_imbalance = False):
        x_train, y_train, x_test, y_test = self._train_test()

        if self.number_of_samples_to_use is not None:
            n_samples = self.get_n_samples(self.number_of_samples_to_use)
            n_labels = np.repeat(self.class_removed, self.number_of_samples_to_use)

            x_train = np.concatenate((x_train, n_samples))
            y_train = np.concatenate((y_train, n_labels))
            x_train, y_train = self._unison_shuffle(x_train, y_train)

        if self.generated_data is not None:
            # assert self.number_of_samples_to_use is not None
            x_train = np.concatenate((x_train, self.generated_data))
            y_train = np.concatenate((y_train, np.repeat(self.class_removed, len(self.generated_data))))
            x_train, y_train = self._unison_shuffle(x_train, y_train)

        if self.generated_data is not None or self.number_of_samples_to_use is not None:
            x_test = np.concatenate((x_test, self.x_class_removed_test))
            y_test = np.concatenate((y_test, self.y_class_removed_test))
            x_test, y_test = self._unison_shuffle(x_test, y_test)
            
        if fix_class_imbalance:
            x_train, y_train = self._balance(x_train, y_train)

        return x_train, self._one_hot_encode(y_train), x_test, self._one_hot_encode(y_test)

    # supply class_index if want only data for this class
    def into_evaluate(self):
        if self.class_removed is not None and (
                self.number_of_samples_to_use is not None or self.generated_data is not None):
            return self._unison_shuffle(
                np.concatenate((self.x_test, self.x_class_removed_test)),
                np.concatenate((self.y_test_one_hot, self._one_hot_encode(self.y_class_removed_test))))
        else:
            return self.x_test, self.y_test_one_hot

    def into_evaluate_one_class(self, class_index):
        if self.class_removed is not None and class_index == self.class_removed:
            y_test_sub = self.y_class_removed_test
            X_test_sub = self.x_class_removed_test
        else:
            is_in_class_subset = self.y_test[:] == class_index
            y_test_sub = self.y_test[is_in_class_subset]
            X_test_sub = self.x_test[is_in_class_subset]
        return X_test_sub, self._one_hot_encode(y_test_sub)

    def set_removed_class(self, class_index, verbose=True):
        self.__init__(use_data_subset=self.use_data_subset, 
                      use_features=self.use_features,
                      class_removed=class_index)

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

            self.y_train_one_hot = self._one_hot_encode(self.y_train)
            self.y_test_one_hot = self._one_hot_encode(self.y_test)

        train_unique, train_counts = np.unique(self.y_train, return_counts=True)
        test_unique, test_counts = np.unique(self.y_test, return_counts=True)

        if verbose:
            if class_index is not None:
                print("Removed class # %d" % class_index)
            else:
                print("Reset to use all classes")
            print("current number of examples per class -- train:\n",
                  dict(zip(train_unique, train_counts)))
            print("\ncurrent number of examples per class -- test:\n",
                  dict(zip(test_unique, test_counts)))


    # number of samples from the removed class
    def set_number_of_samples_to_use(self, n):
        assert n >= 0
        if self.class_removed is None:
            raise Exception("must run d.set_removed_class(...) before setting number of samples to use")

        if n == 0 or n is None:
            self.number_of_samples_to_use = None
        else:
            self.number_of_samples_to_use = n

        self.y_train_one_hot = self._one_hot_encode(self.y_train)
        self.y_test_one_hot = self._one_hot_encode(self.y_test)

    def set_generated_data(self, generated_data):
        assert self.class_removed != None
        # if self.number_of_samples_to_use is None:
        #     raise Exception("must run d.set_number_of_samples_to_use(...) before setting generated_data")
        self.generated_data = generated_data
        self.y_train_one_hot = self._one_hot_encode(self.y_train)
        self.y_test_one_hot = self._one_hot_encode(self.y_test)

    def get_n_samples(self, n):
        if self.class_removed is None:
            raise Exception(
                "must run d.set_removed_class(...) in order to get n samples")

        sample_indexes = self.permutation_for_sample_fetch[:n]
        return self.x_class_removed_train[sample_indexes]

    def get_num_classes(self):
        return self.n_classes

    def get_generated_data_stub(self):
        if self.class_removed is None:
            raise Exception(
                "must run d.set_removed_class(...) in order to get get_generated_data_stub")
        return self.x_class_removed_train[-50:]

    # this method overrides the disease removed paramater!
    def to_low_shot_dataset(self, verbose=False, class_removed=None):
        self.__init__(use_data_subset=False, use_features=self.use_features, class_removed=class_removed)
        if verbose:
            print("Note class_removed paramater was overriden. re-Run d.set_removed_class() if needed")
        
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

    def _processed_data(self, class_removed=None):
        if self.use_features is True:
            return self._load_features(class_removed)
        else:
            return self._load_raw_data()

    def _load_features(self, class_removed=None):
        if class_removed is None:
            read_path = read_features_from_all_classes_path(dataset=self.name)
        else:
            read_path = read_features_from_all_classes_but_one_path(dataset=self.name,
                                                                    class_removed=class_removed)

        dict = joblib.load(read_path)
        features = dict['features']
        labels = dict['labels']

        x_train = features[:self.train_size]
        y_train = labels[:self.train_size]

        x_test = features[self.train_size:]
        y_test = labels[self.train_size:]

        return (x_train, y_train), (x_test, y_test)

    def _one_hot_encode(self, y):
        n_rows, n_cols = len(y), self.n_classes
        enc = np.zeros((n_rows, self.n_classes))
        enc[np.arange(n_rows), y] = 1.0
        if self.class_removed is not None and self.number_of_samples_to_use is None and self.generated_data is None:
            enc = np.delete(enc, [self.class_removed], axis=1)
        return enc

    def _one_hot_decode(self, one_hot_y):
        dec = np.argmax(one_hot_y, axis=1)
        if len(one_hot_y[0]) == self.n_classes:
            return dec
        else:
            offset_fix = np.greater_equal(dec, self.class_removed)
            dec = np.add(dec, offset_fix)
            return dec

    def predictions_to_labels(self, preds):  # preds is an array of distributions
        return self._one_hot_decode(preds)

    def _train_test(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def _unison_shuffle(self, x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def _subset(self, x, y):
        assert len(x) == len(y)
        n = len(x)
        subset_size = round(FRACTION_FOR_SUBSET * n)
        return x[:subset_size], y[:subset_size]

    def _features_and_labels(self):
        x_train, y_train, x_test, y_test = self._train_test()
        return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

    # expects integer labels
    def _balance(self, x, y):
        labels = np.unique(y)
        n_labels = len(labels)
        n_samples_per_label = round((float(len(y)) / (n_labels - 1.0)))

        map_label_to_features = {label : x[y[:] == label] for label in labels}

        balanced_x, balanced_y = [], []
        
        for label in labels:
            for i in range(n_samples_per_label):
                balanced_x.append(random.choice(map_label_to_features[label]))
                balanced_y.append(label)

        return self._unison_shuffle(np.array(balanced_x), np.array(balanced_y))


