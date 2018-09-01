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


def main(category, classifier_epochs, generator_epochs, lambda_percentage, hidden_size, smart_category):
    data_object = MnistData(use_features=False, class_removed=category)
    data_object.set_removed_class(category)

    classifier = MnistClassifier(use_features=False, epochs=classifier_epochs)
    classifier.fit(*data_object.into_fit())
    classifier.set_trainability(is_trainable=False)

    g = LowShotGenerator(classifier.model,
                         data_object,
                         epochs=generator_epochs,
                         hidden_size=hidden_size,
                         λ=lambda_percentage / 100)

    n_real_examples = 3
    n_new = 3
    examples = data_object.get_n_samples(n=n_real_examples)

    vectors, titles = [], []

    for example in examples:
        new_examples, triplets = g.generate_from_samples([example],
                                                         n_total=2,
                                                         smart_category=smart_category,
                                                         smart_centroids=False,
                                                         return_triplets=True)

        original_shape = example.shape
        new_examples = new_examples.reshape((1,) + original_shape)

        for (ϕ, c1a, c2a), new_example in zip(triplets, new_examples):
            vectors += [ϕ.reshape(original_shape), c1a.reshape(original_shape), c2a.reshape(original_shape),
                        new_example]
            titles += ['ϕ', r'$c^{a}_{1}$', r'$c^{a}_{2}$', 'Generated example']

    add_images_to_figure(vectors=vectors, titles=titles, n_cols=4)

    name = 'visualize_{0}.ce_{1}.ge_{2}.l_{3}.hs_{4}.s_{5}.png'.format(category, classifier_epochs, generator_epochs,
                                                                       lambda_percentage, hidden_size, smart_category)
    plt.savefig(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('category', help='what category to exclude', type=int)
    parser.add_argument('-ce', '--classifier_epochs', help='n epochs for classifier training', type=int, default=2)
    parser.add_argument('-ge', '--generator_epochs', help='n epochs for generator training', type=int, default=2)
    parser.add_argument('-l', '--lambda_percentage', help='lambda hyperparameter value in percentage', type=int,
                        default=95)
    parser.add_argument('-hs', '--hidden_size', help='hidden size of generator', type=int, default=256)
    parser.add_argument('-s', '--smart_category', help='whether to use smart category or not', action='store_true')

    args = parser.parse_args()

    main(args.category, args.classifier_epochs, args.generator_epochs, args.lambda_percentage, args.hidden_size,
         args.smart_category)
