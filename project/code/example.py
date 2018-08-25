from generator import LowShotGenerator
from classifier import Classifier
from callbacks import CloudCallback
from train import get_trained_classifier_and_data

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

# all_diseases = ['Atelectasis',
#                 'Cardiomegaly',
#                 'Consolidation',
#                 'Edema',
#                 'Effusion',
#                 'Emphysema',
#                 'Fibrosis',
#                 'Hernia',
#                 'Infiltration',
#                 'Mass',
#                 'No Finding',
#                 'Nodule',
#                 'Pleural_Thickening',
#                 'Pneumonia',
#                 'Pneumothorax']

all_diseases = list(range(15))


def main(disease_name, n_clusters, n_files, λ, test):
    if test:
        n_clusters = 20
        n_files = 12
        λ = 10.

    data_obj = du.get_processed_data(n_files)
    le = du.get_label_encoder(data_obj)
    disease = le.transform(disease_name)

    unused_diseases = [disease]
    diseases_to_remove = [disease_name]

    print('Unused diseases: {0}'.format(', '.join(diseases_to_remove)))
    diseases = [d for d in all_diseases if d not in unused_diseases]

    classifier, X_train, X_test, y_train, y_test = get_trained_classifier_and_data(diseases_to_remove, n_files=n_files)
    classifier.toggle_trainability()  # make the classifier non-trainable

    quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters, categories=diseases, n_files=n_files)

    lsg_name = 'lsg_f.{0}_c.{1}_w.{2}'.format(n_files, n_clusters, '.'.join(unused_diseases))
    lsg = LowShotGenerator(classifier.model, quadruplets_data, λ=λ, name=lsg_name)

    # callback = CloudCallback(True, config.slack_url, config.stop_url, config.slack_channel, name=lsg_name)
    # lsg.fit(callbacks=[callback])

    lsg.fit()

    unused_data = collect.load_quadruplets(n_clusters=n_clusters, categories=unused_diseases, n_files=n_files)
    quadruplets, centroids, cat_to_vectors, cat_to_onehots, original_shape = unused_data

    n_examples = min(len(vecs) for cat, vecs in cat_to_vectors.items() if cat not in diseases_to_remove)

    n_samples = 10
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
    parser.add_argument('disease', help='disease to experiment on')
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-c', '--n_clusters', help='number of clusters', type=int, default=20)
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')
    parser.add_argument('-l', '--λ', help='set λ regularization parameter', type=float, default=10.)

    args = parser.parse_args()

    main(args.disease, args.n_clusters, args.n_files, args.λ, args.test)
