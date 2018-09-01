from generator import LowShotGenerator
from mnist_classifier import *
from mnist_data import *
# from cifar_classifier import *
# from cifar_data import *

import matplotlib.pyplot as plt
import pandas as pd
import collect
import argparse

plt.switch_backend('agg')


def add_image_to_figure(vector, width=28, height=28):
    fig = plt.imshow(np.reshape(vector, (height, width)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def add_images_to_figure(vectors, titles, n_cols, width=28, height=28):
    n_rows = len(vectors) / n_cols if len(vectors) % n_cols == 0 else (len(vectors) / n_cols) + 1
    for i, (vector, title) in enumerate(zip(vectors, titles)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(title)
        add_image_to_figure(vector.astype(float), height, width)

    plt.tight_layout()


def main(category):
    data_object = MnistData(use_features=False, class_removed=category)
    data_object.set_removed_class(category)

    classifier = MnistClassifier(use_features=False)
    classifier.fit(*data_object.into_fit())
    classifier.set_trainability(is_trainable=False)

    g = LowShotGenerator(classifier.model, data_object)

    n_real_examples = 1
    n_new = 3
    n_examples = data_object.get_n_samples(n=n_real_examples)

    new_examples = g.generate_from_samples(n_examples,
                                           n_total=n_new + n_real_examples,
                                           smart_category=False,
                                           smart_centroids=False)

    add_images_to_figure(vectors=n_examples + new_examples,
                         titles=['Source'] + ['Hallucination #{0}'.format(i) for i in range(len(new_examples))],
                         n_cols=n_new+1)

    plt.savefig('visualize_{0}.png'.format(category))


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('category', help='what category to exclude')

    args = parser.parse_args()

    main(args.category)
