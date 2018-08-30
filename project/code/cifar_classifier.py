from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from classifier import *

IMG_ROWS, IMG_COLS = 32, 32
N_CHANNELS = 3

class Cifar10Classifier(Classifier):
    def __init__(self, use_features = False, batch_size = 128, epochs = 125):
        super().__init__(use_features = use_features, 
                         batch_size = batch_size, 
                         epochs = epochs)

    def _cnn(self, n_classes):
        weight_decay = 1e-4
        model = Sequential()
        if self.use_features == False:
            model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=self.input_shape))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.2))
             
            model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.3))
             
            model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.4))
            model.add(Flatten(name='features'))
            model.add(Dense(n_classes, activation='softmax'))
        else:
            model.add(Dense(n_classes, activation='softmax', input_shape=self.input_shape))
        
        self.model = model
        self._compile()
        
        return model

    def _compile(self):
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
