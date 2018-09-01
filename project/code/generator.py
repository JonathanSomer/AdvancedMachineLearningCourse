from keras.models import Model, load_model
from keras.layers import Dense, Input, Reshape
from keras.optimizers import SGD
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from itertools import product
from scipy.spatial.distance import cosine

from mnist_classifier import *
from mnist_data import *

import keras.backend as K

import numpy as np
import data_utils as du
import os
import collect
import config


class LowShotGenerator(object):
    def __init__(self, trained_classifier, data_object, n_layers=3, hidden_size=256,
                 batch_size=128, epochs=10, activation='relu', callbacks=[],
                 name='LowShotGenerator', λ=.95, lr=.1, momentum=.9, decay=1e-4, n_clusters=30):

        self.n_clusters = n_clusters
        self.novel_category = data_object.class_removed
        if self.novel_category is not None:
            self.dataset_name = '{0}_{1}'.format(data_object.name, self.novel_category)

        if not data_object.use_features:
            self.dataset_name = 'raw_{0}'.format(self.dataset_name)

        quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters,
                                                    categories='all',  # we work with dataset_name so it's fine
                                                    dataset_name=self.dataset_name)

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

        self.generated_samples_category_mem = {}
        self.generated_samples_centroids_mem = {}

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

        self.fit()

        # self.weights_file_path = du.read_model_path('{0}'.format(self.name))
        # if os.path.exists(self.weights_file_path):
        #     self.model.load_weights(self.weights_file_path)

        # self.weights_file_path = du.generator_model_path(self.name)
        # if os.path.exists(self.weights_file_path):
        #     self.model.load_weights(self.weights_file_path)
        #     print('Loaded generator weights from file')
        # else:
        #     self.fit()

    @staticmethod
    def build(trained_classifier, original_shape, input_dim, generator_output_dim, n_layers, hidden_size,
              activation, λ=10., lr=.1, momentum=.9, decay=1e-4):
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
        :param smart: whether to use smart selection or not
        :return: list of hallucinated feature vectors G([ϕ, c1a , c2a]) of size n_new
        """
        ϕ = ϕ.flatten()

        def select_category():
            available_categories = list(self.centroids.keys())
            return np.random.choice(available_categories)

        def select_couples_of_centroids(category):
            available_centroids = self.centroids[category]
            idxs = np.random.choice(len(available_centroids), n_new * 2)
            chosen_centroids = available_centroids[idxs]
            c1as, c2as = np.split(chosen_centroids, 2)
            return zip(c1as, c2as)

        category = select_category()
        X = [np.concatenate((ϕ, c1a, c2a)) for c1a, c2a in select_couples_of_centroids(category)]

        return self.generator.predict(np.array(X))

    def generate_from_samples(self, samples, n_total=20, smart_category=False, smart_centroids=False,
                              return_triplets=False):
        n_new = n_total - len(samples)
        n_new_per_sample = n_new // len(samples)

        def select_categories():
            if smart_category in (True, 'smart'):
                print('Selecting classification-closest category to {0}'.format(self.novel_category))
                preds = self.trained_classifier.predict(np.array(samples))
                # y_preds = np.argmax(preds, axis=1)
                y_preds = self.data_object.predictions_to_labels(preds)
                categories, counts = np.unique(y_preds, return_counts=True)
                cnt = dict(zip(categories, counts))
                selected = max(categories, key=lambda c: cnt[c])
                return [selected for _ in samples]

            else:  # random choice
                print('Selecting category randomally')
                available_categories = list(self.centroids.keys())
                return np.random.choice(available_categories, len(samples))

        def select_couples_of_centroids(categories):
            if smart_centroids in ('1', 'cosine_both'):
                print('Selecting cosine-closest centroids as sources')
                print('Selecting cosine-furthest centroids as targets')
                c1as, c2as = [], []
                for i, ϕ in enumerate(samples):
                    category = categories[i]
                    available_centroids = self.centroids[category]
                    cosine_argsort = np.argsort([cosine(ϕ, c) for c in available_centroids])
                    sorted_centroids = available_centroids[cosine_argsort]
                    c1as.append(sorted_centroids[:n_new_per_sample])
                    c2as.append(sorted_centroids[-n_new_per_sample:])
                c1as, c2as = np.concatenate(c1as), np.concatenate(c2as)

            elif smart_centroids in ('2', 'cosine'):
                print('Selecting cosine-closest centroids as sources')
                print('Selecting target centroids randomally')
                c1as, c2as = [], []
                for i, ϕ in enumerate(samples):
                    category = categories[i]
                    available_centroids = self.centroids[category]
                    cosine_argsort = np.argsort([cosine(ϕ, c) for c in available_centroids])
                    sorted_centroids = available_centroids[cosine_argsort]
                    c1as.append(sorted_centroids[:n_new_per_sample])
                    target_options = sorted_centroids[n_new_per_sample:]
                    idxs = np.random.choice(len(target_options), n_new_per_sample)
                    c2as.append(target_options[idxs])
                c1as, c2as = np.concatenate(c1as), np.concatenate(c2as)

            elif smart_centroids in ('3', 'norm_both'):
                print('Selecting norm-closest centroids as sources (NearestNeighbors style)')
                print('Selecting norm-furthest centroids as targets (NearestNeighbors style)')
                c1as, c2as = [], []
                for i, ϕ in enumerate(samples):
                    category = categories[i]
                    available_centroids = self.centroids[category]
                    cosine_argsort = np.argsort([np.linalg.norm(ϕ - c, ord=2) for c in available_centroids])
                    sorted_centroids = available_centroids[cosine_argsort]
                    c1as.append(sorted_centroids[:n_new_per_sample])
                    c2as.append(sorted_centroids[-n_new_per_sample:])
                c1as, c2as = np.concatenate(c1as), np.concatenate(c2as)

            elif smart_centroids in ('4', 'norm'):
                print('Selecting norm-closest centroids as sources (NearestNeighbors style)')
                print('Selecting target centroids randomally')
                c1as, c2as = [], []
                for i, ϕ in enumerate(samples):
                    category = categories[i]
                    available_centroids = self.centroids[category]
                    cosine_argsort = np.argsort([np.linalg.norm(ϕ - c, ord=2) for c in available_centroids])
                    sorted_centroids = available_centroids[cosine_argsort]
                    c1as.append(sorted_centroids[:n_new_per_sample])
                    target_options = sorted_centroids[n_new_per_sample:]
                    idxs = np.random.choice(len(target_options), n_new_per_sample)
                    c2as.append(target_options[idxs])
                c1as, c2as = np.concatenate(c1as), np.concatenate(c2as)

            else:  # random choice
                print('Selecting centroids randomally')
                c1as, c2as = [], []
                for category in categories:
                    available_centroids = self.centroids[category]
                    idxs = np.random.choice(len(available_centroids), n_new_per_sample * 2)
                    _c1as, _c2as = np.split(available_centroids[idxs], 2)
                    c1as += [_c1as]
                    c2as += [_c2as]

                return np.concatenate(c1as), np.concatenate(c2as)

            return np.array(c1as), np.array(c2as)

        samples = np.array([ϕ.flatten() for ϕ in samples])

        categories = select_categories()
        c1as, c2as = select_couples_of_centroids(categories)
        triplets = list(zip(np.repeat(samples, n_new_per_sample, axis=0), c1as, c2as))

        X = [np.concatenate((ϕ, c1a, c2a)) for ϕ, c1a, c2a in triplets]
        preds = self.generator.predict(np.array(X))

        return (preds, triplets) if return_triplets else preds

    @staticmethod
    def benchmark(Classifier, data_object, dataset_name, n_clusters, λ, n_new=100, epochs=10, hidden_size=256):
        """
        :param Classifier: Classifier class (for creating classifiers)
        :param dataset_name: the dataset name of the dataset for the Classifier, mnist or xray
        :param n_clusters: number of clusters to use
        :param λ: lambda parameter
        :param n_new: number of new examples to test accuracy on
        :param epochs: number of epochs to fit with
        :param hidden_size: size of the hidden layers of the generator
        :return: dict mapping category to accuracy
        """
        use_features = 'raw' not in dataset_name
        categories = collect.get_categories(dataset_name)

        # quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters,
        #                                             categories=categories,  # can be 'all' as well
        #                                             dataset_name=dataset_name)
        # quadruplets, cat_to_centroids, cat_to_vectors, original_shape = quadruplets_data

        data_object.set_removed_class(None)
        all_classifier = Classifier(use_features=use_features)
        all_classifier.fit(*data_object.into_fit())

        losses, accs, cat_to_n_unique = {}, {}, {}

        for category in categories:  # iterate on categories to test on each one of them (category is the int label)
            data_object.set_removed_class(category)
            all_but_one_classifier = Classifier(use_features=use_features)
            all_but_one_classifier.fit(*data_object.into_fit())
            all_but_one_classifier.set_trainability(is_trainable=False)

            g = LowShotGenerator(all_but_one_classifier.model,
                                 data_object,
                                 λ=λ,
                                 epochs=epochs,
                                 hidden_size=hidden_size,
                                 n_clusters=n_clusters)

            n_real_examples = 10
            samples = data_object.get_n_samples(n=n_real_examples)
            n_new_per_example = n_new // n_real_examples
            new_examples = np.concatenate([g.generate(ϕ, n_new_per_example) for ϕ in samples])

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
    def benchmark_single(Classifier, DataClass, dataset_name, n_clusters=30, λ=.95, n_new=100, epochs=2,
                         hidden_size=256, classifier_epochs=1, smart_category=False, smart_centroids=False):
        """
        runs a benchmark test on the one category from the given dataset.
        :param Classifier: Classifier class (for creating classifiers i.e. MnistClassifier)
        :param DataClass: DataClass class (for creating data objects. i.e. MnistData)
        :param dataset_name: The dataset name of the wanted dataset to run benchmark on. i.e. mnist_5
        :param n_clusters: number of clusters to use
        :param λ: lambda parameter
        :param n_new: number of new examples to test accuracy on
        :param epochs: number of epochs to fit with
        :param hidden_size: size of the hidden layers of the generator
        :return: dict mapping category to accuracy
        """
        use_features = 'raw' not in dataset_name
        # categories = collect.get_categories(dataset_name)

        # raw_mnist_1 or mnist_1 etc
        try:
            split = dataset_name.split('_')
            if len(split) == 3:
                dataset_name, category_to_exclude = split[1:]
            else:
                dataset_name, category_to_exclude = split
        except ValueError:
            raise ValueError('Given dataset does not fit.')

        category_to_exclude = int(category_to_exclude)

        # get all the quadruplets without the excluded category
        # quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters,
        #                                             categories=categories,
        #                                             dataset_name=dataset_name)

        data_object = DataClass(use_features=use_features, class_removed=category_to_exclude)

        all_classifier = Classifier(use_features=use_features, epochs=classifier_epochs)
        all_classifier.fit(*data_object.into_fit())

        data_object.set_removed_class(category_to_exclude)
        all_but_one_classifier = Classifier(use_features=use_features, epochs=classifier_epochs)
        all_but_one_classifier.fit(*data_object.into_fit())
        all_but_one_classifier.set_trainability(is_trainable=False)

        g = LowShotGenerator(all_but_one_classifier.model,
                             data_object,
                             λ=λ,
                             epochs=epochs,
                             hidden_size=hidden_size,
                             n_clusters=n_clusters)

        n_real_examples = 10
        n_new_per_example = n_new // n_real_examples
        n_examples = data_object.get_n_samples(n=n_real_examples)
        # new_examples = np.concatenate([g.generate(ϕ, n_new_per_example) for ϕ in n_examples])

        new_examples = g.generate_from_samples(n_examples,
                                               n_total=n_new + n_real_examples,
                                               smart_category=smart_category,
                                               smart_centroids=smart_centroids)

        n_unique = len(np.unique(new_examples, axis=0))

        data_object.set_removed_class(None)
        X_new = new_examples
        y_new = data_object._one_hot_encode(np.repeat(category_to_exclude, n_new))

        print('Testing the ALL classifier on generated data:')
        loss, acc = all_classifier.evaluate(X_new, y_new)

        # print('{0} => accuracy: {1}, unique new examples: {2}/{3}'.format(category_to_exclude, acc, n_unique, n_new))
        print('Unique new examples: {0}/{1}'.format(n_unique, n_new))

        return loss, acc, n_unique

    @staticmethod
    def cross_validate(Classifier, data_object, dataset_name, n_clusters=30, n_new=100, epochs=2, test=False):
        import requests

        def slack_update(msg):
            print(msg)
            payload = {'message': msg, 'channel': config.slack_channel}
            requests.post(config.slack_url, json=payload)

        if test:
            hidden_sizes = (4, 8,)
            lambdas = (.95,)
            epochs = 1
            n_clusters = 10
        else:
            hidden_sizes = (16, 32, 64, 128, 256, 512)
            lambdas = (.05, .1, .25, .5, .75, .9, .95)
            # hidden_sizes = (32, 64, 128, 256, 512)
            # lambdas = (.95,)

        avg_losses, avg_accs = {}, {}

        def _benchmark(_hs, _λ):
            args = (Classifier, data_object, dataset_name, n_clusters, _λ, n_new, epochs, _hs)
            return LowShotGenerator.benchmark(*args)

        for hs, λ in product(hidden_sizes, lambdas):
            losses, accs, cat_to_n_unique = _benchmark(hs, λ)
            avg_losses[hs, λ] = sum(losses.values()) / len(losses)
            avg_accs[hs, λ] = sum(accs.values()) / len(accs)

            txt = 'category {0},\tloss = {1}\tacc = {2}\tunique = {3}'
            rows = [txt.format(k, losses[k], accs[k], cat_to_n_unique[k]) for k in sorted(losses.keys())]
            msg = '*hidden_size = {0}, lambda = {1} [{2} clusters, {3} epochs, {5}]*\navg loss = {6}, avg acc = {7}\n```{4}```'.format(
                hs,
                λ,
                n_clusters,
                epochs,
                '\n'.join(rows),
                dataset_name,
                avg_losses[hs, λ],
                avg_accs[hs, λ]
            )
            slack_update(msg)

        hs, λ = min(avg_losses, key=lambda k: avg_losses[k])
        txt = '*Best hidden_size = {0}, lambda = {1}*\navg loss = {2}, avg acc = {3} [{4} clusters, {5} epochs, {6}]'
        slack_update(txt.format(hs, λ, avg_losses[hs, λ], avg_accs[hs, λ], n_clusters, epochs, dataset_name))

        return hs, λ
