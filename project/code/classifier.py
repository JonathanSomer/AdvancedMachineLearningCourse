from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

class Classifier(object):
	    
    def __init__(self, use_features = False, batch_size = 128, epochs = 1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_features = use_features
        self.model = None
        self.trainable = True

    # must get one hot encoded labels
    def fit(self, x_train, y_train, x_test, y_test):
        assert len(y_train[0]) == len(y_test[0])
        assert x_train[0].shape == x_test[0].shape
        self.input_shape = x_train[0].shape
        self.model = self._cnn(n_classes = len(y_train[0]))
        self.model.fit(x_train, y_train,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=1,
          validation_data=(x_test, y_test))

    def evaluate(self, x_test, y_test, verbose=True):
        if self.model is None:
            raise Exception("must run cls.fit(...) in order to evaluate the model")

        x_test = np.array([np.reshape(x, self.input_shape) for x in x_test])
        score =  self.model.evaluate(x_test, y_test, verbose=1)
        if verbose:
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
        return score

    def predict(self, x):
        return self.model.predict(x)

    def set_trainability(self, is_trainable):
        self.trainable = is_trainable
        for layer in self.model.layers:
            layer.trainable = self.trainable

        if self.trainable:
            print('Classifier was set to trainable!')
        else:
            print('Classifier was set to NOT trainable!')

        self._compile()
