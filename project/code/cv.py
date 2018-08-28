from generator import LowShotGenerator
from mnist_classifier import *
from mnist_data import *

import argparse

Classifiers = {'mnist': MnistClassifier}
data_obj_getters = {'mnist': MnistData}


def main(dataset_name, n_clusters, test):
    LowShotGenerator.cross_validate(Classifiers[dataset_name],
                                    data_obj_getters[dataset_name](),
                                    dataset_name,
                                    n_clusters=n_clusters,
                                    test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='what dataset to use', default='mnist')  # xray is the second one
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=40)
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')

    args = parser.parse_args()

    main(args.dataset, args.n_clusters, args.test)