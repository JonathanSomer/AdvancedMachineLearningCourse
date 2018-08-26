from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling3D, AveragePooling2D
from keras.optimizers import Adam, SGD
from sklearn.utils.class_weight import compute_class_weight

import config
import numpy as np


class Classifier(object):
    def __init__(self, n_classes=15, model_weights_file_path=None, trainable=True, n_epochs=100):
        self.n_epochs = n_epochs
        self.trainable = trainable

        if config.resnet_version_performs_pooling:
            self.model = Classifier.new_model_no_pooling(n_classes, trainable=trainable)
        else:
            self.model = Classifier.new_model(n_classes, trainable=trainable)

        if model_weights_file_path is not None:
            self.model.load_weights(model_weights_file_path)
            print('Loaded classifier weights from a saved model')

        self.compile()

    def toggle_trainability(self):
        if self.trainable:
            print('Classifier is now non-trainable!')
        else:
            print('Classifier is now trainable!')

        self.trainable = not self.trainable

        for layer in self.model.layers[1:]:  # first layer is input
            layer.trainable = self.trainable

        self.compile()

    @staticmethod
    def new_model(n_classes, trainable=True):
        a = Input(shape=(7, 7, 2048,))
        b = GlobalMaxPooling2D(trainable=trainable)(a)
        b = Dense(n_classes, activation='softmax', name='classifier', trainable=trainable)(b)
        return Model(a, b)

    @staticmethod
    def new_model_no_pooling(n_classes, trainable=True):
        a = Input(shape=(2048,))
        b = Dense(n_classes, activation='softmax', name='classifier', trainable=trainable)(a)
        return Model(a, b)

    @staticmethod
    def get_optimizer():
        # return SGD()
        # return Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        return Adam(lr=1e-4)

    @staticmethod
    def get_class_weights(y_train, as_array=False):  # expects a one hot encoded y_train array
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        if as_array:
            return class_weights
        class_weights_dict = dict(enumerate(class_weights))
        return class_weights_dict

    def compile(self):
        optimizer = Classifier.get_optimizer()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    # supply file_path if want to save model to file
    def fit(self, X_train, y_train, model_weights_file_path=None, callbacks=[]):
        class_weight = Classifier.get_class_weights(y_train)
        self.model.fit(X_train, y_train, batch_size=50, epochs=self.n_epochs, verbose=1, validation_split=0.1, callbacks=callbacks, class_weight=class_weight)

        if model_weights_file_path:
            self.model.save(model_weights_file_path)

        return self.model

    # returns an array with [loss, accuracy]
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, batch_size=32, verbose=1)

    def predict(self, X_test):
        return self.model.predict(X_test)
