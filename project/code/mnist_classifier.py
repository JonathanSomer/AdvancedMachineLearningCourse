from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from classifier import *

IMG_ROWS, IMG_COLS = 28, 28

# acheives 98% accuracy within 2 epochs
class MnistClassifier(Classifier):
    def __init__(self, use_features = False, batch_size = 128, epochs = 1):
        super().__init__(use_features = use_features, 
                         batch_size = batch_size, 
                         epochs = epochs)

    def _cnn(self, n_classes):
        model = Sequential()
        if not self.use_features:
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu', name="features"))
            model.add(Dropout(0.5))
            model.add(Dense(n_classes, activation='softmax', name="last"))
        else:
            model.add(Dropout(0.5, input_shape=self.input_shape))
            model.add(Dense(n_classes, activation='softmax', name="last"))

        self.model = model
        self._compile()
        
        return model

    def _compile(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
