from keras.models import Model, load_model
from keras.layers import Dense, Input, Reshape, Lambda
from itertools import combinations
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from sklearn.externals import joblib

import numpy as np
import data_utils as du
import os


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class LowShotGenerator(object):
    def __init__(self, linear_classifier, dataset, n_layers=3, n_clusters=100, n_cpus=4, hidden_size=512,
                 batch_size=100, n_epochs=10, activation='relu', n_examples=None, callbacks=[],
                 name='lsg', force_rebuild=False):
        """
        dataset is a dict mapping base class to its feature vectors + dict mapping from cat to its onehot encoding
        n_examples is k in the paper: the minimum number of examples per novel category
        # TODO: maybe need to add/remove another Dense layer (the paper says it should 3 layers so it's ambigous)
        """

        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_cpus = n_cpus
        # self.dataset = dataset
        self.cat_to_vectors, self.cat_to_onehots, self.original_shape = dataset
        self.hidden_size = hidden_size
        self.W = self.linear_classifier = linear_classifier
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.name = name

        # n_examples is the maximum number to generate per class
        if n_examples:
            self.n_examples = n_examples
        else:
            self.n_examples = min(len(vs) for vs in self.cat_to_vectors.values())

        # load quadruplets & centroids if exists
        self.quadruplets_file_path = du.pickle_path('{0}-{1}c-{2}h-quadruplets'.format(name, n_clusters, hidden_size))
        if os.path.exists(self.quadruplets_file_path) and not force_rebuild:
            print('Loading quadruplets and centroids.')
            loaded = joblib.load(self.quadruplets_file_path)
            self.quadruplets, self.centroids = loaded['quadruplets'], loaded['centroids']
        else:
            self.quadruplets, self.centroids = self.create_quadruplets()

            print('Saving quadruplets and centroids.')
            joblib.dump({'quadruplets': self.quadruplets, 'centroids': self.centroids}, self.quadruplets_file_path)

        # load train data if exists
        self.train_data_file_path = du.pickle_path('{0}-{1}c-{2}h-train_data'.format(name, n_clusters, hidden_size))
        if os.path.exists(self.train_data_file_path) and not force_rebuild:
            print('Loading samples.')
            loaded = joblib.load(self.train_data_file_path)
            self.x_train, self.y_train = loaded['x_train'], loaded['y_train']
            self.original_shape = loaded['original_shape']
        else:
            print('Creating samples.')
            self.x_train = np.array([np.concatenate((c1a, c1b, c2b)) for ((c1a, c2a, c1b, c2b), _) in self.quadruplets])
            self.y_train = {'generator': np.array([np.array(c2a) for ((c1a, c2a, c1b, c2b), cat) in self.quadruplets]),
                            'classifier': np.array([self.cat_to_onehots[cat] for (_, cat) in self.quadruplets])}

            print('Saving samples.')
            joblib.dump({'x_train': self.x_train, 'y_train': self.y_train, 'original_shape': self.original_shape},
                        self.train_data_file_path)

        # load model & generator if exists
        self.model_path = du.model_path('{0}-{1}c-{2}h'.format(name, n_clusters, hidden_size))
        self.generator_path = du.model_path('{0}_generator'.format(name))
        if os.path.exists(self.model_path) and os.path.exists(self.generator_path) and not force_rebuild:
            print('Loading model & generator.')
            self.model = load_model(self.model_path)
            self.generator = load_model(self.generator_path)
        else:
            self.model, self.generator = self.build()
            print(self.model.summary())

            # self.fit()

            # print('Saving model & generator after fitting.')
            # self.model.save(self.model_path)

    def create_quadruplets(self, verbose=0):
        clusters = {}
        for category, X in self.cat_to_vectors.items():
            print('Running KMeans to get {0} clusters for "{1}".'.format(self.n_clusters, category))
            kmeans = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_cpus, verbose=verbose).fit(X)
            clusters[category] = kmeans

        print('Creating quadruplets (2 pairs of 2 centroids).')
        quadruplets = []
        for a, b in combinations(clusters, 2):
            for c1a, c2a in combinations(clusters[a].cluster_centers_, 2):
                min_dist, quadruplet, category = float('inf'), None, None
                for c1b, c2b in combinations(clusters[b].cluster_centers_, 2):
                    dist = cosine(c1a - c2a, c1b - c2b)
                    if dist < min_dist:
                        min_dist, quadruplet, category = dist, (c1a, c2a, c1b, c2b), a

                c1a, c2a, c1b, c2b = quadruplet
                if cosine_similarity(c1a - c2a, c1b - c2b) > 0:
                    quadruplets.append((quadruplet, a))

        centroids = {category: cluster.cluster_centers_ for category, cluster in clusters.items()}

        return quadruplets, centroids

    def build(self, λ=.5):  # TODO: For testing, I chose λ = .5 arbitrarily. We need to figure it out.
        print('Building model.')

        input_dim = len(self.x_train[0])
        generator_output_dim = len(self.y_train['generator'][0])

        curr = inputs = Input(shape=(input_dim,))

        # hidden layers creation, as I understand the paper it should be 3 - 1 = 2 in our case
        for _ in range(self.n_layers-1):
            curr = Dense(self.hidden_size, activation=self.activation)(curr)

        generator_output = Dense(generator_output_dim, activation=self.activation, name='generator')(curr)

        # classifier_wrapper is dummy layer for mapping the losses by giving a name to the classifier model
        reshape = Reshape(self.original_shape)(generator_output)
        classifier = self.linear_classifier.model(reshape)
        classifier_wrapper = Lambda(lambda x: x, name='classifier')(classifier)

        model = Model(inputs=inputs, outputs=[generator_output, classifier_wrapper])
        generator = Model(inputs=inputs, outputs=generator_output)

        losses = {'generator': 'mse',
                  'classifier': 'categorical_crossentropy'}

        weights = {'generator': λ,
                   'classifier': 1.}

        model.compile(loss=losses,
                      loss_weights=weights,
                      optimizer='adam',
                      metrics=['accuracy'])

        return model, generator

    def fit(self):
        print('Fitting model.')
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.n_epochs,
                       callbacks=self.callbacks)

    def generate(self, ϕ, n_new=1):
        """
        :param ϕ: real example for some category
        :param n_new: number of new examples to return. should be <= self.n_examples
        :return: list of hallucinated feature vectors G([ϕ, c1a , c2a]) of size n_new
        """
        X = []
        for _ in range(min(n_new, self.n_examples)):
            sample_category = np.random.choice(list(self.cat_to_vectors.keys()))
            centroids = self.centroids[sample_category]

            idx = np.random.choice(len(centroids), 2)
            c1a, c2a = centroids[idx]

            x = np.concatenate((ϕ, c1a, c2a))
            X.append(x)

        return self.generator.predict(np.array(X))
