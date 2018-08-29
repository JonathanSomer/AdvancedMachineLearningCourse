from keras.models import Model, load_model
from keras.layers import Dense, Input, Reshape
from keras.optimizers import SGD
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from itertools import product

from mnist_classifier import *
from mnist_data import *

import keras.backend as K

import numpy as np
import data_utils as du
import os
import collect
import config


class LowShotGenerator(object):
    def __init__(self, trained_classifier, quadruplets_data, data_object, n_layers=3, hidden_size=512,
                 batch_size=128, epochs=10, activation='relu', n_examples=None, callbacks=[],
                 name='LowShotGenerator', λ=10., lr=.1, momentum=.9, decay=1e-4):

        self.n_layers = n_layers
        self.quadruplets, self.centroids, self.cat_to_vectors, self.original_shape = quadruplets_data
        self.categories = list(self.cat_to_vectors.keys())
        self.n_classes = len(self.categories)
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
        self.data_object = data_object

        x_train, y_classifier, y_generator = [], [], []

        for category, quadruplets in self.quadruplets.items():
            for (c1a, c2a, c1b, c2b) in quadruplets:
                x_train.append(np.concatenate((c1a, c1b, c2b)))
                y_generator.append(np.array(c2a))
                y_classifier.append(category)

        y_classifier = data_object._one_hot_encode(y_classifier)

        self.x_train = np.array(x_train)
        self.y_train = {'generator': np.array(y_generator),
                        'classifier': np.array(y_classifier)}

        self.input_dim = len(self.x_train[0])
        self.generator_output_dim = len(self.y_train['generator'][0])

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

        # self.weights_file_path = du.read_model_path('{0}'.format(self.name))
        # if os.path.exists(self.weights_file_path):
        #     self.model.load_weights(self.weights_file_path)

        # self.weights_file_path = du.generator_model_path(self.name)
        # if os.path.exists(self.weights_file_path):
        #     self.model.load_weights(self.weights_file_path)
        #     print('Loaded generator weights from file')
        # else:
        #     self.fit()

        self.fit()

    @staticmethod
    def build(trained_classifier, original_shape, input_dim, generator_output_dim, n_layers, hidden_size,
              activation,
              λ=10., lr=.1, momentum=.9, decay=1e-4):
        # verify that the trained classifier is not trainable
        n_trainable_params = np.sum(K.count_params(p) for p in set(trained_classifier.trainable_weights))
        if n_trainable_params > 0:
            raise ValueError('The given classifier is trainable.')

        curr = inputs = Input(shape=(input_dim,))

        for _ in range(n_layers - 1):
            curr = Dense(hidden_size, activation=activation)(curr)

        curr = generator_output = Dense(generator_output_dim, activation=activation, name='generator')(curr)

        if original_shape != (generator_output_dim,):
            curr = Reshape(original_shape)(generator_output)

        # the input of the trained_classifier is the output the generator
        classifier = Model(trained_classifier.inputs, trained_classifier.outputs, name='classifier')
        classifier_output = classifier(curr)
        # classifier = trained_classifier(curr)

        model = Model(inputs=inputs, outputs=[generator_output, classifier_output])
        generator = Model(inputs=inputs, outputs=generator_output)

        loss = {'generator': 'mse',
                'classifier': 'categorical_crossentropy'}

        loss_weights = {'generator': λ,
                        'classifier': 1. - λ}

        optimizer = SGD(lr=lr, momentum=momentum, decay=decay)

        model.compile(loss=loss,
                      loss_weights=loss_weights,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # print('Generator summary:')
        # print(generator.summary())
        # print('\nWhole model summary:')
        # print(model.summary())
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

        # weights_file_path = du.write_model_path('{0}'.format(self.name))
        # self.model.save(weights_file_path)

        # self.model.save(self.weights_file_path)

    def generate(self, ϕ, n_new=1):
        """
        :param ϕ: "seed" example for some novel category
        :param n_new: number of new examples to return. should be less than self.n_examples
        :return: list of hallucinated feature vectors G([ϕ, c1a , c2a]) of size n_new
        """
        X = []
        for _ in range(n_new):
            # this is a list of lists, each list is centroids of cat
            centroids_all_categories = list(self.centroids.values())
            idx = np.random.choice(len(centroids_all_categories))

            category_centroids = centroids_all_categories[idx]

            idx = np.random.choice(len(category_centroids), 2)
            c1a, c2a = category_centroids[idx]

            # print(ϕ.shape)
            # print(c1a.shape)
            x = np.concatenate((ϕ.flatten(), c1a, c2a))
            X.append(x)

        return self.generator.predict(np.array(X))

    @staticmethod
    def get_generated_features(classifier, novel_category_label, seed_examples_of_novel_category, n_clusters, λ,
                               n_new=20, dataset_name='xray'):
        name = '{3}.{0}.{1}_lambda.{2}_clusters'.format(novel_category_label, λ, n_clusters, dataset_name)

        if classifier.trainable:
            classifier.toggle_trainability()

        all_labels = list(range(10 if dataset_name == 'mnist' else 15))
        trained_categories = [d for d in all_labels if d != novel_category_label]

        quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters,
                                                    categories=trained_categories,
                                                    dataset_name=dataset_name)

        generator = LowShotGenerator(classifier.model,
                                     quadruplets_data,
                                     λ=λ,
                                     name=name)

        n_new_per_example = (n_new - len(seed_examples_of_novel_category)) // len(seed_examples_of_novel_category)
        new_examples = [generator.generate(ϕ, n_new=n_new_per_example) for ϕ in seed_examples_of_novel_category]

        return np.concatenate(new_examples)

    @staticmethod
    def benchmark(Classifier, data_object, dataset_name, n_clusters, λ, n_new=100, epochs=10, hidden_size=512):
        """
        :param Classifier: Classifier class (for creating classifiers)
        :param dataset_name: the dataset name of the dataset for the Classifier, mnist or xray
        :param n_clusters: number of clusters to use
        :param λ: lambda parameter
        :param n_new: number of new examples to test accuracy on
        :param epochs: number of epochs to fit with
        :return: dict mapping category to accuracy
        """
        categories = list(range(collect.dataset_to_n_categories[dataset_name]))

        quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters,
                                                    categories=categories,  # can be 'all' as well
                                                    dataset_name=dataset_name)
        quadruplets, cat_to_centroids, cat_to_vectors, original_shape = quadruplets_data

        # X = np.concatenate(tuple(cat_to_vectors.values()))
        # y = np.concatenate(tuple([c] * len(v) for c, v in cat_to_vectors.items()))  # y as ints
        # onehot_encode_all_func, y_onehot = LowShotGenerator.onehot_encode(y)

        # all_classifier = Classifier((n_classes=len(categories), n_epochs=epochs))
        # all_classifier.fit(X, y_onehot)
        # all_classifier.toggle_trainability()

        data_object.set_removed_class(None)
        all_classifier = Classifier()
        all_classifier.fit(*data_object.into_fit())

        # masks = defaultdict(list)
        # for i, c in enumerate(y):
        #     masks[c] += [i]

        losses, accs, cat_to_n_unique = {}, {}, {}

        for category in categories:  # iterate on categories to test on each one of them (category is the int label)
            # setup boolean mask
            # mask = np.ones(len(X), dtype=bool)
            # mask[masks[category]] = False

            # X_category, X_rest = X[~mask], X[mask]
            # y_category, y_rest = y[~mask], X[mask]

            # _, y_rest_onehot = LowShotGenerator.onehot_encode(y_rest)

            # all_but_one_classifier = Classifier(n_classes=len(categories) - 1, n_epochs=epochs)
            # all_but_one_classifier.fit(X_rest, y_rest_onehot)

            data_object.set_removed_class(category)
            all_but_one_classifier = Classifier()
            all_but_one_classifier.fit(*data_object.into_fit())
            all_but_one_classifier.set_trainability(is_trainable=False)

            g = LowShotGenerator(all_but_one_classifier.model,
                                 quadruplets_data,
                                 data_object,
                                 λ=λ,
                                 epochs=epochs,
                                 hidden_size=hidden_size)

            n_real_examples = 10
            n_new_per_example = n_new // n_real_examples
            n_examples = data_object.get_n_samples(n=n_real_examples)
            new_examples = np.concatenate([g.generate(ϕ, n_new_per_example) for ϕ in n_examples])

            cat_to_n_unique[category] = n_unique = len(np.unique(new_examples, axis=0))

            data_object.set_removed_class(None)
            X_new = new_examples
            y_new = data_object._one_hot_encode(np.repeat(category, n_new))

            print('Testing the ALL classifier on generated data:')
            loss, acc = all_classifier.evaluate(X_new, y_new)
            losses[category], accs[category] = loss, acc

            print('{0} => accuracy: {1}, unique new examples: {2}/{3}'.format(category, acc, n_unique, n_new))

        return losses, accs, cat_to_n_unique

    @staticmethod
    def onehot_encode(y, n_classes=None):
        yy = y.reshape(-1, 1)
        encoder = OneHotEncoder(n_values=n_classes) if n_classes else OneHotEncoder()
        encoder.fit(yy)
        one_hot_labels = encoder.transform(yy).toarray()

        def encode_func(z):
            return encoder.transform(z.reshape(-1, 1)).toarray()

        return encode_func, one_hot_labels

    @staticmethod
    def cross_validate(Classifier, data_object, dataset_name, n_clusters=40, n_new=100, epochs=2, test=False):
        import requests
        from concurrent.futures import ProcessPoolExecutor

        def slack_update(msg):
            print(msg)
            payload = {'message': msg, 'channel': config.slack_channel}
            requests.post(config.slack_url, json=payload)

        if test:
            hidden_sizes = (4, 8,)
            lambdas = (.5, .75, .9, .95)
            epochs = 1
            n_clusters = 10
        else:
            # hidden_sizes = (16, 128, 512)
            # lambdas = (.05, .25, .5, .75, .95)
            hidden_sizes = (32, 64, 128, 256, 512)
            lambdas = (.95,)

        avg_losses, avg_accs = {}, {}

        def _benchmark(_hs, _λ):
            args = (Classifier, data_object, dataset_name, n_clusters, _λ, n_new, epochs, _hs)
            return LowShotGenerator.benchmark(*args)

        with ProcessPoolExecutor() as executor:
            # for hs, λ in product(hidden_sizes, lambdas):
            #     losses, accs, cat_to_n_unique = _benchmark(hs, λ)

            for hs, λ, (losses, accs, cat_to_n_unique) in zip(hidden_sizes, lambdas,
                                                              executor.map(_benchmark, hidden_sizes, lambdas)):
                avg_losses[hs, λ] = sum(losses.values()) / len(losses)
                avg_accs[hs, λ] = sum(accs.values()) / len(accs)

                txt = 'category {0},\tloss = {1}\tacc = {2}\tunique = {3} [%d clusters, %d epochs]' % (n_clusters,
                                                                                                       epochs)
                rows = [txt.format(k, losses[k], accs[k], cat_to_n_unique[k]) for k in sorted(losses.keys())]
                msg = '*hidden_size = {0}, lambda = {1}*\n```{2}```'.format(hs, λ, '\n'.join(rows))
                slack_update(msg)

        hs, λ = min(avg_losses, key=lambda k: avg_losses[k])
        txt = '*Best hidden_size = {0}, lambda = {1}*\navg loss = {2}, avg acc = {3} [%d clusters, %d epochs]' % (
            n_clusters,
            epochs)
        slack_update(txt.format(hs, λ, avg_losses[hs, λ], avg_accs[hs, λ]))

        return hs, λ
