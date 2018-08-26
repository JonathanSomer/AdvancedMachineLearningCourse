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
ALL_DISEASE_NAMES = ['Atelectasis',
                    'Cardiomegaly',
                    'Consolidation',
                    'Edema',
                    'Effusion',
                    'Emphysema',
                    'Fibrosis',
                    'Hernia',
                    'Infiltration',
                    'Mass',
                    'No Finding',
                    'Nodule',
                    'Pleural_Thickening',
                    'Pneumonia',
                    'Pneumothorax']
TOTAL_SAMPLES_TO_GENERATE = 20

DEFAULT_MODEL_FILE = 'model'

# returns an absolute pickle/model path. local_data_dir should be configured in config.py
# local_data_dir is the data directory and should contain the following directories: datasets, pickles, models, images
# for instance: '~/amldata

def read_pickle_path(name):
    return os.path.join(config.local_data_dir, 'pickles', 'read', '{0}.pickle'.format(name))


def write_pickle_path(name):
    return os.path.join(config.local_data_dir, 'pickles', 'write', '{0}.pickle'.format(name))


def read_model_path(name=DEFAULT_MODEL_FILE):
    return os.path.join(config.local_data_dir, 'models', 'read', '{0}.h5'.format(name))

def is_model_file_exists(name=DEFAULT_MODEL_FILE):
    return os.path.isfile(read_model_path(name))

def write_model_path(name):
    return os.path.join(config.local_data_dir, 'models', 'write', '{0}.h5'.format(name))


def generator_model_path(name):
    return os.path.join(config.local_data_dir, 'models', 'generators', '{0}.h5'.format(name))


def read_plot_path(name):
    return os.path.join(config.local_data_dir, 'plots', 'read', '{0}.png'.format(name))


def write_plot_path(name):
    return os.path.join(config.local_data_dir, 'plots', 'write', '{0}.png'.format(name))


def read_result_path(name):
    return os.path.join(config.local_data_dir, 'results', 'read', '{0}.json'.format(name))


def write_result_path(name):
    return os.path.join(config.local_data_dir, 'results ', 'write', '{0}.json'.format(name))


def image_data_pickle_name(i):
    return 'image_data_{0}'.format(i)


def images_directory(i):
    return os.path.join(config.local_data_dir, 'images', 'images_{0}/'.format(i))


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
        file_path = read_pickle_path(image_data_pickle_name(i))
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


def onehot_encode(y, n_classes=None):
    yy = y.reshape(-1, 1)
    enc = OneHotEncoder(n_values=n_classes) if n_classes else OneHotEncoder()
    enc.fit(yy)
    one_hot_labels = enc.transform(yy).toarray()
    return one_hot_labels

def get_label_encoder(data_obj):
    le = preprocessing.LabelEncoder()
    le.classes_ = data_obj['label_encoder_classes']
    return le


def new_get_train_test_split_without_disease(X, y, disease, data_obj):
    X_no_disease, y_no_disease = remove_diseases(X, y, [disease], data_obj)
    X_train, X_test, y_train, y_test = new_get_train_test_split(X_no_disease, y_no_disease, test_size=0.1)
    return X_train, X_test, y_train, y_test

def new_get_train_test_split(X, y, test_size=0.1):
    test_mask = get_choose_n_mask(len(y), round(test_size*len(y)))
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test

def new_onehot_encode(y, disease_indexes_removed=[]):
    yy = y.reshape(-1, 1)
    enc = OneHotEncoder(n_values=N_CLASSES)
    enc.fit(yy)
    one_hot_labels = np.array(enc.transform(yy).toarray())
    one_hot_labels = np.delete(one_hot_labels, disease_indexes_removed, axis=1)
    return one_hot_labels

def get_all_disease_samples_and_rest(X,y, disease_index, data_obj, n=TOTAL_SAMPLES_TO_GENERATE):
    include_disease_mask = y[:] == disease_index
    disease_X = X[include_disease_mask]
    disease_y = y[include_disease_mask]
    n_mask = get_choose_n_mask(len(disease_y), n)
    return disease_X[n_mask], disease_y[n_mask], disease_X[~n_mask], disease_y[~n_mask]

def add_disease_to_test_data(X_test, y_test, disease_X, disease_y):
    X_test_with_disease = np.concatenate((X_test, disease_X))
    y_test_with_disease = np.concatenate((y_test, disease_y))
    return unison_shuffle(X_test, y_test)

def add_n_samples_to_train_data(X_train, y_train, all_samples_features, all_samples_labels, n):
    n_samples_features = all_samples_features[:n+1]
    n_labels = all_samples_labels[:n+1]
    X_train_with_samples = np.concatenate((X_train, n_samples_features))
    y_train_with_samples = np.concatenate((y_train, n_labels))
    X_train_with_samples, y_train_with_samples = unison_shuffle(X_train_with_samples, y_train_with_samples)
    return X_train_with_samples, y_train_with_samples, n_samples_features

def add_generated_data_to_train_data(X_train, y_train, generated_features, generated_features_label):
    X_train_with_generated_data = np.concatenate((X_train, generated_features))
    y_train_with_generated_data = np.concatenate((y_train, np.repeat(generated_features_label, len(generated_features))))
    return unison_shuffle(X_train_with_generated_data, y_train_with_generated_data)

def get_train_test_split(X, y, test_size=0.1):
    onehot_labels = onehot_encode(y)

    test_mask = get_choose_n_mask(len(y), round(test_size*len(y)))
    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test

def disease_name_to_index(disease_name, data_obj):
    le = get_label_encoder(data_obj)
    return le.transform([disease_name])[0]

def disease_index_to_name(disease_index, data_obj):
    le = get_label_encoder(data_obj)
    return le.inverse_transform([disease_index])[0]

def remove_diseases(X, y, diseases_to_remove, data):
    black_list = [disease_name_to_index(name, data) for name in diseases_to_remove]

    include = ~np.isin(y, black_list)
    return X[include], y[include]

def split_by_disease(X, y, disease_index, data):
    include_disease = y[:] == disease_index
    return X[~include_disease], y[~include_disease], X[include_disease], y[include_disease]

def get_train_test_split_without_disease(X, y, disease, data_obj):
    X_no_disease, y_no_disease = remove_diseases(X, y, [disease], data_obj)
    X_train, X_test, y_train, y_test = get_train_test_split(X_no_disease, y_no_disease, test_size=0.1)
    return X_train, X_test, y_train, y_test

# training data includes n samples of the disease
def get_train_test_split_with_n_samples_of_disease(X, y, disease, data_obj, n):
    disease_index = disease_name_to_index(disease, data_obj) if type(disease) == str else disease

    X_no_disease, y_no_disease, X_only_disease, y_only_disease = split_by_disease(X, y, disease_index, data_obj)
    X_train, X_test, y_train, y_test = train_test_split(X_no_disease, y_no_disease)

    choose_n_mask = get_choose_n_mask(len(y_only_disease), n)
    n_samples_features, n_samples_integer_labels = X_only_disease[choose_n_mask], y_only_disease[choose_n_mask]

    X_train = np.concatenate((X_train, n_samples_features))
    y_train = np.concatenate((y_train, n_samples_integer_labels))

    X_test = np.concatenate((X_test, X_only_disease[~choose_n_mask]))
    y_test = np.concatenate((y_test, y_only_disease[~choose_n_mask]))

    X_train, y_train = unison_shuffle(X_train, onehot_encode(y_train, n_classes=N_CLASSES))
    X_test, y_test = unison_shuffle(X_test, onehot_encode(y_test, n_classes=N_CLASSES))

    return X_train, X_test, y_train, y_test, n_samples_features, n_samples_integer_labels

# uses the train and test data from get_train_test_split_with_n_samples_of_disease()
# and the generated data
def get_train_test_with_generated_data(X_train, X_test, y_train, y_test, generated_features, generated_data_label):
    X_train = np.concatenate((X_train, generated_features))
    y_train = np.concatenate((y_train, onehot_encode(np.repeat(generated_data_label, len(generated_features)), n_classes=N_CLASSES)))

    X_train, y_train = unison_shuffle(X_train, y_train)
    return X_train, X_test, y_train, y_test

def unison_shuffle(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def get_choose_n_mask(mask_size, n):
    choose_n_mask = np.array([True]*n + [False]*(mask_size - n))
    np.random.shuffle(choose_n_mask)
    return choose_n_mask

def all_disease_names_except(name):
    names = list(ALL_DISEASE_NAMES)
    names.remove(name)
    return names

def shallow_no_finding(X, y, data):
    le = preprocessing.LabelEncoder()
    le.classes_ = data['label_encoder_classes']
    black_list = le.transform(['No Finding'])
    include = ~np.isin(y, black_list[:10000])
    return X[include], y[include]


def get_single_disease_images_dataframe():
    df = pd.read_csv(DATA_DIRECTORY + 'Data_Entry_2017.csv')
    single_disease_images = df.loc[~df['Finding Labels'].str.contains('\|')]
    return single_disease_images


def file_names_by_directory(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


def get_resnet_model():
    return ResNet50(weights='imagenet', include_top=False)


# in order to extract image features into pickle files run:
# df = get_single_disease_images_dataframe()
# model = get_resnet_model()
# extract_image_features(model, df, test_run=False)

# link to kaggle data: https://www.kaggle.com/nih-chest-xrays/data
# note: must have images_1,....images_12 directories of images from kaggle in local_data_dir/images/ directory
#       must have Data_Entry_2017.csv from kaggle in data/ directory
def extract_image_features(model=None, single_disease_images_dataframe=None, test_run=True):
    if model is None:
        model = get_resnet_model()

    if single_disease_images_dataframe is None:
        single_disease_images_dataframe = get_single_disease_images_dataframe()

    le = preprocessing.LabelEncoder()
    le.fit(single_disease_images_dataframe['Finding Labels'])
    label_encoder_classes = le.classes_

    cnt = 0

    num_files = 2 if test_run else 13
    for file_index in range(1, num_files):

        file_indexes = []
        image_names = []
        features = []
        int_labels = []

        directory = images_directory(file_index)
        names_of_images_in_directory = pd.Series(file_names_by_directory(directory))
        single_disease_images_in_directory_bool = names_of_images_in_directory.isin(
            single_disease_images_dataframe['Image Index'])
        single_disease_images_in_directory = names_of_images_in_directory[single_disease_images_in_directory_bool]

        for image_name in single_disease_images_in_directory:

            image_path = directory + image_name
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_features = model.predict(x)

            text_label = \
            single_disease_images_dataframe.loc[single_disease_images_dataframe['Image Index'] == image_name][
                'Finding Labels'].values[0]

            file_indexes.append(file_index)
            image_names.append(image_name)
            features.append(image_features)
            int_labels.extend(le.transform([text_label]))

            cnt += 1
            if cnt % 10 == 0:
                print("working on file #%d , handled %d images" % (file_index, cnt))
            if test_run and cnt > 20:
                break

        data = {
            'file_indexes': file_indexes,
            'image_names': image_names,
            'features': features,
            'int_labels': int_labels,
            'label_encoder_classes': label_encoder_classes
        }

        d = pickle.dumps(data)
        with open(write_pickle_path(image_data_pickle_name(file_index)), 'wb') as f:
            for i in range(0, len(d), MAX_BYTES):
                f.write(d[i:i + MAX_BYTES])


def write_large_object_to_file(obj, path):
    d = pickle.dumps(obj)
    max_bytes = 2 ** 31 - 1
    with open(path, 'wb') as f:
        for i in range(0, len(d), max_bytes):
            f.write(d[i:i + max_bytes])


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

        if diseases_to_remove:
            X, y = remove_diseases(X, y, diseases_to_remove, data)

        cat_to_vectors = defaultdict(list)
        original_shape = X[0].shape
        for i, x, yy in zip(range(len(X)), X, y):
            cat_to_vectors[yy].append(x.flatten())

        cat_to_vectors = {y: np.array(X) for y, X in cat_to_vectors.items()}

        return cat_to_vectors, original_shape
