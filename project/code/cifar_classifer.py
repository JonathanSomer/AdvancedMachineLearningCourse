from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

IMG_ROWS, IMG_COLS = 32, 32
N_CHANNELS = 3

class Cifar10Classifier(object):
    def __init__(self, batch_size = 128, epochs = 3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = self._input_shape()
        self.model = None
        self.trainable = True

    def _input_shape(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (N_CHANNELS, IMG_ROWS, IMG_COLS)
        else:
            input_shape = (IMG_ROWS, IMG_COLS, N_CHANNELS)
        return input_shape

    def _mnist_cnn(self, n_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))
        self.model = model
        self._compile()
        
        return model

    # must get one hot encoded labels
    def fit(self, x_train, y_train, x_test, y_test):
        assert len(y_train[0]) == len(y_test[0])
        assert x_train[0].shape == x_test[0].shape
        self.model = self._mnist_cnn(n_classes = len(y_train[0]))
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

    def _compile(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
