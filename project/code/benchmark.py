from generator import LowShotGenerator
from mnist_classifier import *
from mnist_data import *
from cifar_data import *
from itertools import product

import argparse
import matplotlib.pyplot as plt
import pandas as pd

Classifiers = {'mnist': MnistClassifier}

data_obj_getters = {'mnist': MnistData,
                    'cifar10': Cifar10Data}


def main(dataset, category, n_clusters, generator_epochs, classifier_epochs, n_new, to_cross_validate, smart_category,
         smart_centroids, plot):
    categories = range(data_obj_getters[dataset]().get_num_classes()) if category == 'all' else [category]
    results = {}

    for category in categories:
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
                                                     epochs=generator_epochs,
                                                     classifier_epochs=classifier_epochs,
                                                     smart_category=smart_category,
                                                     smart_centroids=smart_centroids)

        if to_cross_validate:
            raise NotImplementedError('Still does not work.')
            hidden_sizes = (64, 128, 256, 512)
            lambdas = (.1, .25, .5, .75, .9, .95)

            losses, accs, n_uniques = {}, {}, {}

            for hs, λ in product(hidden_sizes, lambdas):
                losses[hs, λ], accs[hs, λ], n_uniques[hs, λ] = _benchmark(hs, λ)

        else:
            loss, acc, n_unique = _benchmark(256, .95)
            results[category] = {'loss': loss, 'acc': acc, 'n_unique': n_unique}

        if plot:
            results['avg'] = {'loss': np.average([v['loss'] for k, v in results.items()]),
                              'acc': np.average([v['acc'] for k, v in results.items()]),
                              'n_unique': np.average([v['n_unique'] for k, v in results.items()])}

            df = pd.DataFrame.from_dict(results)
            df.plot()
            plt.tight_layout()
            plt.savefig('./benchmark.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='what dataset to use')
    parser.add_argument('category', help='what category to benchmark on', type=str)
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=30)
    parser.add_argument('-ge', '--generator_epochs', help='number of epcohs to train the generator with', type=int,
                        default=2)
    parser.add_argument('-ce', '--classifier_epochs', help='number of epcohs to train the classifier with', type=int,
                        default=1)
    parser.add_argument('-n', '--n_new', help='num of new examples to create and evaluate', type=int, default=100)
    parser.add_argument('-cv', '--cross_validate', help='whether to do a cross validation', action='store_true')
    parser.add_argument('-sca', '--smart_category', help='wether to generate with samrt category', action='store_true')
    parser.add_argument('-sce', '--smart_centroids', help='wether to generate with samrt centroids',
                        type=str, default='none')
    parser.add_argument('-p', '--plot', help='whether to plot the results', action='store_true')

    args = parser.parse_args()

    main(args.dataset, args.category, args.n_clusters, args.generator_epochs, args.classifier_epochs, args.n_new,
         args.cross_validate, args.smart_category, args.smart_centroids, args.plot)
