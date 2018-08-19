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
import pickle

N_CLASSES = 15
MAX_BYTES = 2**31 - 1 # max size of block which pickle can write in one shot
DATA_DIRECTORY = '../data/'
READ_ONLY_PROCESSED_DATA = DATA_DIRECTORY + 'processed_data/'


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
    'file_indexes' : [],
    'image_names' : [],
    'features' : [],
    'int_labels' : [],
    'label_encoder_classes' : [] 
    }

    for i in range(1, min(13, num_files_to_fetch_data_from + 1)):
        print("fetching data from file #%d" % i)
        file_path = READ_ONLY_PROCESSED_DATA + 'image_data_' + str(i) + '.pickle'
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

def images_directory(i):
    return DATA_DIRECTORY + 'images_' + str(i) + '/'

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
# extract_image_features(model, df)

# link to kaggle data: https://www.kaggle.com/nih-chest-xrays/data
# note: must have images_1,....images_12 directories of images from kaggle in data/ directory
#       must have Data_Entry_2017.csv from kaggle in data/ directory
def extract_image_features(model=None, single_disease_images_dataframe=None):

    if model is None:
        model = get_resnet_model()

    if single_disease_images_dataframe is None:
        single_disease_images_dataframe = get_single_disease_images_dataframe()

    le = preprocessing.LabelEncoder()
    le.fit(single_disease_images_dataframe['Finding Labels'])
    label_encoder_classes = le.classes_

    cnt = 0

    for file_index in range(1,2):

        file_indexes = []
        image_names = []
        features = []
        int_labels = []

        directory = images_directory(file_index)
        names_of_images_in_directory = pd.Series(file_names_by_directory(directory))
        single_disease_images_in_directory_bool = names_of_images_in_directory.isin(single_disease_images_dataframe['Image Index'])
        single_disease_images_in_directory = names_of_images_in_directory[single_disease_images_in_directory_bool]

        for image_name in single_disease_images_in_directory:


            image_path = directory + image_name
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            image_features = model.predict(x)

            text_label = single_disease_images_dataframe.loc[single_disease_images_dataframe['Image Index'] == image_name]['Finding Labels'].values[0]

            file_indexes.append(file_index)
            image_names.append(image_name)
            features.append(image_features)
            int_labels.extend(le.transform([text_label]))


            cnt += 1
            if cnt % 10 == 0:
                print("working on file #%d , handled %d images" % (file_index, cnt))

        data = {
        'file_indexes' : file_indexes,
        'image_names' : image_names,
        'features' : features,
        'int_labels' : int_labels,
        'label_encoder_classes' : label_encoder_classes
        }

        d = pickle.dumps(data)
        with open(DATA_DIRECTORY + 'image_data_' + str(file_index) + '.pickle', 'wb') as f:
            for i in range(0, len(d), MAX_BYTES):
                f.write(d[i:i+MAX_BYTES])


def write_large_object_to_file(obj, path):
    d = pickle.dumps(obj)
    max_bytes = 2**31 - 1
    with open(path, 'wb') as f:
        for i in range(0, len(d), max_bytes):
            f.write(d[i:i+max_bytes])
