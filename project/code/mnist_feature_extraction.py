from mnist_data import *
from mnist_classifier import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.externals import joblib
from data_utils import *

MNIST_FEATURES_PICKLE_NAME = 'mnist_features'

d = MnistData(use_data_subset=False)
cls = MnistClassifier()

# train classifier on all data
cls.fit(*d.into_fit())

predictor = cls.model
extractor = Model(inputs=predictor.input,
                 outputs=predictor.get_layer('features').output)

x, y = d._features_and_labels()
features = extractor.predict(x)

dataset = {'features' : features, 'labels' : y}
joblib.dump(dataset, write_pickle_path(MNIST_FEATURES_PICKLE_NAME))
