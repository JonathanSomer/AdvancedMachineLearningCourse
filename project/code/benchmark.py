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


def plotify(data_dict, title=None, name=None):
    if not name:
        name = 'benchmark'

    df = pd.DataFrame.from_dict(data_dict)
    df.to_pickle('./{0}.pickle'.format(name))

    df = df.transpose()
    df.plot(kind='bar', figsize=(10, 10))

    plt.ylabel('Evaluation accuracy on generated examples')
    plt.xlabel('Removed category')
    plt.xticks(rotation=0)
    plt.legend(loc='best')

    if title:
        plt.title(title)

    plt.tight_layout()

    plt.savefig('./{0}.png'.format(name))


def main(dataset, category, n_clusters, generator_epochs, classifier_epochs, n_new, to_cross_validate_hyper,
         category_selection, centroids_selection, to_plot):
    dataset_key = dataset.replace('raw_', '')
    categories = range(data_obj_getters[dataset]().get_num_classes()) if category == 'all' else [category]

    if category_selection == 'all':
        category_selection_types = ('smart', 'random')
    else:
        category_selection_types = (category_selection,)

    if centroids_selection == 'all':
        centroids_selection_types = ('random', 'cosine_both', 'cosine', 'norm', 'norm_both')
    else:
        centroids_selection_types = (centroids_selection,)

    if to_cross_validate_hyper:
        hidden_sizes = (64, 128, 256, 512)
        lambdas = (.1, .25, .5, .75, .9, .95)
        hidden_sizes_str = lambdas_str = 'all'
    else:
        hidden_sizes = (256,)
        lambdas = (.95,)
        hidden_sizes_str = '256'
        lambdas_str = '0.95'

    name_format = '{0}.category_{1}.category_select_{2}.cendroids_select_{3}.hs_{4}.lambda_{5}'
    name = name_format.format(dataset,
                              categories[0] if len(categories) == 1 else 'all',
                              category_selection_types[0] if len(category_selection_types) == 1 else 'all',
                              centroids_selection_types[0] if len(centroids_selection_types) == 1 else 'all',
                              hidden_sizes_str,
                              lambdas_str)

    all_accs, all_acc_keys = defaultdict(dict), []
    for category, category_selection, centroids_selection, hs, λ in product(categories,
                                                                            category_selection_types,
                                                                            centroids_selection_types,
                                                                            hidden_sizes,
                                                                            lambdas):
        dataset_name = '{0}_{1}'.format(dataset, category)
        acc_key = '{0}_category.{1}_centroids.hs_{2}.λ_{3}'.format(category_selection, centroids_selection, hs, λ)
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

        loss, acc, n_unique = _benchmark(hs, λ)
        all_accs[category][acc_key] = acc * 100

    if to_plot:
        all_accs['avg'] = {acc_key: np.average([v[acc_key] for k, v in all_accs.items()]) for acc_key in
                           all_acc_keys}

        # max_avg_acc = max(list(all_accs['avg'].values()))
        plotify(all_accs, name=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='what dataset to use')
    parser.add_argument('category', help='what category to benchmark on', type=str)
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=30)
    parser.add_argument('-ge', '--generator_epochs', help='number of epcohs to train the generator with', type=int,
                        default=2)
    parser.add_argument('-ce', '--classifier_epochs', help='number of epcohs to train the classifier with', type=int,
                        default=12)
    parser.add_argument('-n', '--n_new', help='num of new examples to create and evaluate', type=int, default=100)
    parser.add_argument('-cv', '--cross_validate_hyper', help='whether to do a cross validation on hyperparams',
                        action='store_true')
    parser.add_argument('-sca', '--category_selection', help='type of category selection [smart/random]', type=str,
                        default='random')
    parser.add_argument('-sce', '--centroids_selection',
                        help='type of centriods selection [cosine_both/cosine/norm_both/norm/random]',
                        type=str, default='random')
    parser.add_argument('-p', '--plot', help='whether to plot the results', action='store_true')

    args = parser.parse_args()

    main(args.dataset, args.category, args.n_clusters, args.generator_epochs, args.classifier_epochs, args.n_new,
         args.cross_validate_hyper, args.category_selection, args.centroids_selection, args.plot)
