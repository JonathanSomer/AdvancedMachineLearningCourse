from generator import LowShotGenerator
from mnist_classifier import *
from mnist_data import *
from cifar_data import *
from itertools import product

import argparse

Classifiers = {'mnist': MnistClassifier}

data_obj_getters = {'mnist': MnistData,
                    'cifar10': Cifar10Data}


def main(dataset, category, n_clusters, epochs, n_new, to_cross_validate):
    dataset_key = dataset.replace('raw_', '')
    dataset_name = '{0}_{1}'.format(dataset, category)

    def _benchmark(_hs, _λ):
        return LowShotGenerator.benchmark_single(Classifiers[dataset_key],
                                                 data_obj_getters[dataset_key],
                                                 dataset_name,
                                                 n_clusters=n_clusters,
                                                 λ=_λ,
                                                 hidden_size=_hs,
                                                 n_new=n_new,
                                                 epochs=epochs)

    if to_cross_validate:
        raise NotImplementedError('Still does not work.')
        hidden_sizes = (64, 128, 256, 512)
        lambdas = (.1, .25, .5, .75, .9, .95)

        losses, accs, n_uniques = {}, {}, {}

        for hs, λ in product(hidden_sizes, lambdas):
            losses[hs, λ], accs[hs, λ], n_uniques[hs, λ] = _benchmark(hs, λ)

    else:
        _benchmark(256, .95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='what dataset to use')
    parser.add_argument('category', help='what category to benchmark on', type=int)
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=30)
    parser.add_argument('-e', '--epochs', help='number of clusters to create', type=int, default=2)
    parser.add_argument('-n', '--n_new', help='num of new examples to create and evaluate', type=int, default=100)
    parser.add_argument('-cv', '--cross_validate', help='where to do a cross validation', action='store_true')

    args = parser.parse_args()

    main(args.dataset, args.category, args.n_clusters, args.epochs, args.n_new, args.cross_validate)
