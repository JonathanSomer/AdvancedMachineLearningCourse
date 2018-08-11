from keras.models import Sequential
from keras.layers import Dense
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine

import numpy as np
import pickle
import os


def pickle_path(name):
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'pickles', name + '.pickle')
    return os.path.abspath(path)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class LowShotGenerator(object):
    def __init__(self, linear_classifier, dataset, n_layers=3, n_clusters=100, n_cpus=4, hidden_size=512,
                 batch_size=100, n_epochs=10, activation='relu', n_examples=None, callbacks=[],
                 name='low-shot-generator', force_rebuild=False):
        """
        dataset is a dict mapping base class to its feature vectors
        n_examples is k in the paper: the minimum number of examples per novel category
        # TODO: maybe need to add/remove another Dense layer (the paper says it should 3 layers so it's ambigous)
        """

        file_path = pickle_path(name)
        if os.path.exists(file_path) and not force_rebuild:
            with open(file_path, "rb") as handle:
                self.__dict__.update(pickle.load(handle))
            return

        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_cpus = n_cpus
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.W = self.linear_classifier = linear_classifier
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.name = name

        if n_examples:
            self.n_examples = n_examples
        else:
            self.n_examples = min(len(vs) for vs in dataset.values())

        self.quadruplets, self.centroids = self.create_quadruplets()

        self.x_train = [c1a + c1b + c2b for ((c1a, c2a, c1b, c2b), cat) in self.quadruplets]
        self.y_train = [(c2a, cat) for ((c1a, c2a, c1b, c2b), cat) in self.quadruplets]

        self.model = self.build()
        self.fit()

    def create_quadruplets(self):
        clusters = {}
        for category, X in self.dataset.items():
            kmeans = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_cpus).fit(X)
            clusters[category] = kmeans

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

        centroids = {cat: cluster.cluster_centers_ for cat, cluster in clusters.items()}

        return quadruplets, centroids

    def build(self):
        input_dim, output_dim = len(self.x_train[0]), len(self.y_train[0])  # TODO: maybe need to use shape
        model = Sequential([Dense(self.hidden_size, activation=self.activation, input_dim=input_dim),
                            Dense(self.hidden_size, activation=self.activation),
                            Dense(output_dim, activation=self.activation)])

        def loss(y_true_with_cat, y_pred):
            # TODO: what λ is?
            λ = 1.
            y_true, category = y_true_with_cat

            loss_mse = mean_squared_error(y_true, y_pred)
            score = self.linear_classifier.evaluate(np.array([y_pred]), np.array([category]))
            loss_cls = np.log(score[0])

            _loss = (λ * loss_mse) + loss_cls
            return _loss

        model.compile(loss=loss, optimizer='adam', metrics=['loss', 'accuracy'])
        return model

    def fit(self):
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.n_epochs,
                       callbacks=self.callbacks)

    def generate(self, 𝟇):
        """
        :param 𝟇: real example for some category
        :return: hallucinated feature vector G([𝟇, c1a , c2a])
        """
        sample_category = np.random.choice(list(self.dataset.keys()))
        c1a, c2a = np.random.choice(self.centroids[sample_category], 2)

        x = np.concatenate((𝟇, c1a, c2a,))
        return self.model.predict(x)

    def save(self, file_path=None):
        if not file_path:
            file_path = pickle_path(self.name)

        with open(file_path, "wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

