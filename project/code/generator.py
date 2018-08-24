from keras.models import Model, load_model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.optimizers import SGD

import keras.backend as K

import numpy as np
import data_utils as du
import os


class LowShotGenerator(object):
    def __init__(self, trained_classifier, quadruplets_data, n_layers=3, hidden_size=512,
                 batch_size=128, epochs=10, activation='relu', n_examples=None, callbacks=[],
                 name='LowShotGenerator', λ=10., lr=.1, momentum=.9, decay=1e-4):

        self.n_layers = n_layers
        self.quadruplets, self.centroids, self.cat_to_vectors, self.cat_to_onehots, self.original_shape = quadruplets_data
        self.hidden_size = hidden_size
        self.W = self.trained_classifier = trained_classifier
        self.activation = activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.name = name
        self.λ = λ
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        x_train, y_classifier, y_generator = [], [], []

        for category, quadruplets in self.quadruplets.items():
            for (c1a, c2a, c1b, c2b) in quadruplets:
                x_train.append(np.concatenate((c1a, c1b, c2b)))
                y_generator.append(np.array(c2a))
                y_classifier.append(self.cat_to_onehots[category])

        self.x_train = np.array(x_train)
        self.y_train = [np.array(y_generator), np.array(y_classifier)]
        # self.y_train = {'generator': np.array(y_generator),
        #                 'classifier': np.array(y_classifier)}

        self.input_dim = len(self.x_train[0])
        self.generator_output_dim = len(self.y_train[0][0])
        # self.generator_output_dim = len(self.y_train['generator'][0])

        # n_examples is the maximum number to generate per class
        # if n_examples:
        #     self.n_examples = n_examples
        # else:
        #     self.n_examples = min(len(vs) for vs in self.cat_to_vectors.values())

        self.model, self.generator = self.build(self.trained_classifier,
                                                self.original_shape,
                                                self.input_dim,
                                                self.generator_output_dim,
                                                self.n_layers,
                                                self.hidden_size,
                                                self.activation,
                                                self.λ,
                                                self.lr,
                                                self.momentum,
                                                self.decay)

        self.weights_file_path = du.read_model_path('{0}'.format(self.name))
        if os.path.exists(self.weights_file_path):
            self.model.load_weights(self.weights_file_path)

    @staticmethod
    def build(trained_classifier, original_shape, input_dim, generator_output_dim, n_layers, hidden_size, activation,
              λ=10., lr=.1, momentum=.9, decay=1e-4):
        # verify that the trained classifier is not trainable
        n_non_trainable_params = np.sum(K.count_params(p) for p in set(trained_classifier.non_trainable_weights))
        if n_non_trainable_params > 0:
            raise ValueError('The given classifier is trainable.')

        curr = inputs = Input(shape=(input_dim,))

        for _ in range(n_layers - 1):
            curr = Dense(hidden_size, activation=activation)(curr)

        curr = generator_output = Dense(generator_output_dim, activation=activation)(curr)

        if original_shape != (generator_output_dim,):
            curr = Reshape(original_shape)(generator_output)

        classifier = Model(trained_classifier.inputs, trained_classifier.outputs)
        classifier_output = classifier(curr)

        model = Model(inputs=inputs, outputs=[generator_output, classifier_output])
        generator = Model(inputs=inputs, outputs=generator_output)

        loss = ['mse', 'categorical_crossentropy']
        loss_weights = [λ, 1]

        optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)

        model.compile(loss=loss,
                      loss_weights=loss_weights,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        print('Generator summary:')
        print(generator.summary())
        print('\nWhole model summary:')
        print(model.summary())
        return model, generator
    
    def fit(self, x_train=None, y_train=None, batch_size=None, epochs=None, callbacks=None):
        print('Fitting generator')

        if not x_train:
            x_train = self.x_train

        if not y_train:
            y_train = self.y_train

        if not batch_size:
            batch_size = self.batch_size

        if not epochs:
            epochs = self.epochs

        if not callbacks:
            callbacks = self.callbacks

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

        weights_file_path = du.write_model_path('{0}'.format(self.name))
        self.model.save(weights_file_path)

        return self.model

    def generate(self, ϕ, n_new=1):
        """
        :param ϕ: "seed" example for some category
        :param n_new: number of new examples to return. should be less than self.n_examples
        :return: list of hallucinated feature vectors G([ϕ, c1a , c2a]) of size n_new
        """
        X = []
        for _ in range(n_new):
            sample_category = np.random.choice(list(self.cat_to_vectors.keys()))
            centroids = self.centroids[sample_category]

            idx = np.random.choice(len(centroids), 2)
            c1a, c2a = centroids[idx]

            x = np.concatenate((ϕ, c1a, c2a))
            X.append(x)

        return self.generator.predict(np.array(X))
