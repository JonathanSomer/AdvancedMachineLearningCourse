from generator import LowShotGenerator
from mnist_classifier import *
from mnist_data import *
from cifar_data import *

import argparse

Classifiers = {'mnist': MnistClassifier}

data_obj_getters = {'mnist': MnistData,
                    'cifar10': Cifar10Data}


def main(dataset, category, n_clusters, epochs, n_new):
    dataset_key = dataset.replace('raw_', '')
    dataset_name = '{0}_{1}'.format(dataset, category)

    LowShotGenerator.benchmark_single(Classifiers[dataset_key],
                                      data_obj_getters[dataset_key],
                                      dataset_name,
                                      n_clusters=n_clusters,
                                      λ=.95,
                                      hidden_size=256,
                                      n_new=n_new,
                                      epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='what dataset to use')
    parser.add_argument('category', help='what category to benchmark on')
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=30)
    parser.add_argument('-e', '--epochs', help='number of clusters to create', type=int, default=2)
    parser.add_argument('-n', '--n_new', help='num of new examples to create and evaluate', type=int, default=100)

    args = parser.parse_args()

    main(args.dataset, args.category, args.n_clusters, args.epochs, args.n_new)