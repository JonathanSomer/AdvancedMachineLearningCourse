import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
from os import listdir
from os.path import isfile, join
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling3D, AveragePooling2D
from keras.optimizers import SGD, Adam
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import defaultdict
import pickle

import config

from keras.utils import to_categorical
import keras.backend as K

N_CLASSES = 15
MAX_BYTES = 2 ** 31 - 1  # max size of block which pickle can write in one shot
DATA_DIRECTORY = '../data/'
READ_ONLY_PROCESSED_DATA = DATA_DIRECTORY + 'processed_data/'


# returns an absolute pickle/model path. local_data_dir should be configured in config.py
# local_data_dir is the data directory and should contain the following directories: datasets, pickles, models
# for instance: '~/amldata

def pickle_path(name):
    return os.path.join(config.local_data_dir, 'pickles', '{0}.pickle'.format(name))


def model_path(name):
    return os.path.join(config.local_data_dir, 'models', '{0}.h5'.format(name))


# BEFORE USING THIS METHOD MAKE SURE YOU HAVE THE NEEDED 'image_data_1.pickle' FILE
# INSIDE THE DIRECTORY '/../data/'
# Use this to fetch the main data object:
# data is partitioned into 12 files so num_files_to_fetch_data_from should be in [1,12]
# The data object contains:
# file_indexes: index of the image directory including the image .png file
# image_names: name of the original .png file
# features: feature vectors of size (1,7,7,2048)
# int_labels: class label
# label_encoder_classes: classes to feed into the LabelEncoder for decoding int labels
def get_processed_data(num_files_to_fetch_data_from):
    data = {
        'file_indexes': [],
        'image_names': [],
        'features': [],
        'int_labels': [],
        'label_encoder_classes': []
    }

    for i in range(1, min(13, num_files_to_fetch_data_from + 1)):
        print("fetching data from file #%d" % i)
        # file_path = DATA_DIRECTORY + 'image_data_' + str(i) + '.pickle'
        file_path = pickle_path('image_data_{0}'.format(i))
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)

        with open(file_path, 'rb') as f:
            for _ in range(0, input_size, MAX_BYTES):
                bytes_in += f.read(MAX_BYTES)

        data_batch = pickle.loads(bytes_in)

        for key in data.keys():
            if i > 1 and key == 'label_encoder_classes':
                continue
            data[key].extend(data_batch[key])

    return data


# returns the features with shape of (7,7,2048)
def get_features_and_labels(data):
    X = np.array(data['features']).squeeze()
    y = np.array(data['int_labels'])
    return X, y


def get_train_test_split(X, y, test_size=0.1):
    yy = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(yy)
    one_hot_labels = enc.transform(yy).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def remove_diseases(X, y, diseases_to_remove, data):
    le = preprocessing.LabelEncoder()
    le.classes_ = data['label_encoder_classes']
    black_list = le.transform(diseases_to_remove)
    include = ~np.isin(y, black_list)
    return X[include], y[include]


# this methods filters diseases_to_remove from data (which is data_obj)
# and then builds a dataset for the LowShotGenerator:
# LowShotGenerator expects to receive as dataset a tuple of:
# 1. dict mapping category (disease) as label string to the category's samsples
# 2. dict mapping category (disease) as label string to the category's onehot encoding
# 3. original_shape of the samples, because the generator expects to get a flattened vector
#    and needs to reshape it to the original shape for the classifier prediction.
#    In our case the original shape is of course (7, 7, 2048), but the function extracts it anyway.
#
# One thing to notice that I used keras.utils.to_categorical in order to onehot encode.
# Probably it's better to onehot encode the same way in both this function and get_train_test_split.
# I leave it for you to decide or we may discuss it together.
def to_low_shot_dataset(data, diseases_to_remove=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X, y = get_features_and_labels(data)

        le = preprocessing.LabelEncoder()
        le.classes_ = data['label_encoder_classes']

        if diseases_to_remove:
            black_list = le.transform(diseases_to_remove)
            include = ~np.isin(y, black_list)
            X, y = X[include], y[include]

        onehot_encoded = to_categorical(y)

        cat_to_vectors = defaultdict(list)
        cat_to_onehots = {}
        original_shape = None
        for i, x, yy in zip(range(len(X)), X, y):
            original_shape = x.shape
            cat = le.inverse_transform(yy)
            cat_to_vectors[cat].append(x.flatten())
            cat_to_onehots[cat] = onehot_encoded[i]

        cat_to_vectors = {y: np.array(X) for y, X in cat_to_vectors.items()}

        return cat_to_vectors, cat_to_onehots, original_shape
