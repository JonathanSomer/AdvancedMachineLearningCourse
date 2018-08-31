from generator import LowShotGenerator
from mnist_classifier import *
from mnist_data import *
from cifar_classifier import *
from cifar_data import *
from itertools import product
from collections import defaultdict

import argparse
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')

Classifiers = {'mnist': MnistClassifier,
               'cifar10': Cifar10Classifier}

data_obj_getters = {'mnist': MnistData,
                    'cifar10': Cifar10Data}


def main(dataset, category, n_clusters, generator_epochs, classifier_epochs, n_new, to_cross_validate,
         category_selection,
         centroids_selection, plot):
    categories = range(data_obj_getters[dataset]().get_num_classes()) if category == 'all' else [category]

    all_accs = defaultdict(dict)

    if category_selection == 'all':
        category_selection_types = ('smart', 'random')
    else:
        category_selection_types = (category_selection,)

    if centroids_selection == 'all':
        centroids_selection_types = ('random', 'cosine_both', 'cosine', 'norm', 'norm_both')
    else:
        centroids_selection_types = (centroids_selection,)

    all_acc_keys = []
    for category, category_selection, centroids_selection in product(categories,
                                                                     category_selection_types,
                                                                     centroids_selection_types):
        dataset_key = dataset.replace('raw_', '')
        dataset_name = '{0}_{1}'.format(dataset, category)
        acc_key = '{0}_category.{1}_centroids'.format(category_selection, centroids_selection)
        all_acc_keys += [acc_key]

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
                                                     smart_category=category_selection,
                                                     smart_centroids=centroids_selection)

        if to_cross_validate:
            raise NotImplementedError('Still does not work.')
            hidden_sizes = (64, 128, 256, 512)
            lambdas = (.1, .25, .5, .75, .9, .95)

            losses, accs, n_uniques = {}, {}, {}

            for hs, λ in product(hidden_sizes, lambdas):
                losses[hs, λ], accs[hs, λ], n_uniques[hs, λ] = _benchmark(hs, λ)

        else:
            loss, acc, n_unique = _benchmark(256, .95)
            all_accs[category][acc_key] = acc * 100

    if plot:
        all_accs['avg'] = {acc_key: np.average([v[acc_key] for k, v in all_accs.items()]) for acc_key in
                           all_acc_keys}

        df = pd.DataFrame.from_dict(all_accs)
        df.to_pickle('./benchmark.pickle')

        df.plot(kind='barh', figsize=(10, 10))
        plt.legend(loc='best')
        plt.tight_layout()

        path_format = './benchmark.png'
        # category_type = 'smart' if smart_category else 'random'
        # centroids_type = 'random' if smart_centroids == 'random' else 'smart{0}'.format(smart_centroids)

        plt.savefig(path_format)


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
    parser.add_argument('-sca', '--category_selection', help='type of category selection [smart/random]', type=str,
                        default='random')
    parser.add_argument('-sce', '--centroids_selection',
                        help='type of centriods selection [cosine_both/cosine/norm_both/norm/random]',
                        type=str, default='random')
    parser.add_argument('-p', '--plot', help='whether to plot the results', action='store_true')

    args = parser.parse_args()

    main(args.dataset, args.category, args.n_clusters, args.generator_epochs, args.classifier_epochs, args.n_new,
         args.cross_validate, args.category_selection, args.centroids_selection, args.plot)
