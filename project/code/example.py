from generator import LowShotGenerator
from classifier import Classifier
from callbacks import CloudCallback
from train import do_train

import numpy as np
import data_utils as du
import collect
import argparse
import config

'''
Number of samples per category:
===============================
0               Hernia    110
1            Pneumonia    322
2                Edema    628
3             Fibrosis    727
4            Emphysema    892
5         Cardiomegaly   1093
6   Pleural_Thickening   1126
7        Consolidation   1310
8                 Mass   2139
9         Pneumothorax   2194
10              Nodule   2705
11            Effusion   3955
12         Atelectasis   4215
13        Infiltration   9547
14          No Finding  60361
'''

all_diseases = ['Atelectasis',
                'Cardiomegaly',
                'Consolidation',
                'Edema',
                'Effusion',
                'Emphysema',
                'Fibrosis',
                'Hernia',
                'Infiltration',
                'Mass',
                'No Finding',
                'Nodule',
                'Pleural_Thickening',
                'Pneumonia',
                'Pneumothorax']


# TODO: change unused_diseases to be 'Hernia', 'Pneumonia', 'Edema', 'Emphysema', 'Fibrosis'
# TODO: add evalutation without 'No Finding'?
# TODO: try to train and evaluate without 'No Finding' at all
# TODO: change it to create a new classifier (and do not load a pretrained one)
def main(n_clusters, n_files, λ, shallow_no_finding, test):
    if test:
        n_clusters = 20
        n_files = 12
        λ = 10.

    unused_diseases = ['Hernia', 'Pneumonia', 'Edema', 'Emphysema', 'Fibrosis']
    diseases_to_remove = list(unused_diseases)
    print('Unused diseases: {0}'.format(', '.join(unused_diseases)))
    diseases = [d for d in all_diseases if d not in unused_diseases]

    # if shallow_no_finding:  # currently it No Finding it entirely
    #     print('No Finding is excluded')
    #     diseases = [d for d in diseases if d != 'No Finding']
    #     diseases_to_remove += ['No Finding']

    classifier, X_train, X_test, y_train, y_test = do_train(diseases_to_remove, n_files=n_files)
    classifier.toggle_trainability()  # make the classifier non-trainable

    quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters, categories=diseases, n_files=n_files)

    # classifier_name = 'classifier_f_{0}_w_{1}'.format(n_files, '.'.join(unused_diseases))
    # classifier = Classifier(model_weights_file_path=du.read_model_path(classifier_name),
    #                         trainable=False)

    lsg_name = 'lsg_f.{0}_c.{1}_w.{2}'.format(n_files, n_clusters, '.'.join(unused_diseases))
    lsg = LowShotGenerator(classifier.model, quadruplets_data, λ=λ, name=lsg_name)

    # callback = CloudCallback(True, config.slack_url, config.stop_url, config.slack_channel, name=lsg_name)
    # lsg.fit(callbacks=[callback])

    lsg.fit()

    unused_data = collect.load_quadruplets(n_clusters=n_clusters, categories=unused_diseases, n_files=n_files)
    quadruplets, centroids, cat_to_vectors, cat_to_onehots, original_shape = unused_data

    n_examples = min(len(ls) for ls in cat_to_vectors.values())

    n_samples = 5
    disease = 'Hernia'
    print('Generating {0} examples from {1} samples of {2}'.format(n_examples, n_samples, disease))

    X_train_disease, X_test_disease = cat_to_vectors[disease][:n_samples], cat_to_vectors[disease][n_samples:]
    X_train_disease = np.concatenate(
        [X_train_disease] + [lsg.generate(ϕ, n_new=n_examples // n_samples) for ϕ in X_train_disease])
    y_train_disease = np.array([cat_to_onehots[disease] for x in X_train_disease])
    y_test_disease = np.array([cat_to_onehots[disease] for x in X_test_disease])

    X_train, y_train = np.concatenate((X_train, X_train_disease)), np.concatenate((y_train, y_train_disease))

    classifier = Classifier(trainable=True)
    classifier.fit(X_train, y_train)

    loss, acc = classifier.evaluate(X_test, y_test)
    print('accuracy for regular diseases is {0}'.format(acc))

    loss, acc = classifier.evaluate(X_test_disease, y_test_disease)
    print('accuracy for novel disease "{0}" is {1}'.format(disease, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-c', '--n_clusters', help='number of clusters', type=int, default=20)
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')
    parser.add_argument('-l', '--λ', help='set λ regularization parameter', type=float, default=10.)
    parser.add_argument('-nf', '--shallow_no_finding', help='whether to shallow No Finding or not', action='store_true')

    args = parser.parse_args()

    main(args.n_clusters, args.n_files, args.λ, args.shallow_no_finding, args.test)
