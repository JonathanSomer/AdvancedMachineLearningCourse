from generator import LowShotGenerator
from classifier import Classifier
from callbacks import CloudCallback

import numpy as np
import data_utils as du
import collect
import argparse
import config

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


def main(n_clusters, n_files, λ, test):
    if test:
        n_clusters = 20
        n_files = 12
        λ = 10.

    unused_diseases = ['Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    print('Unused diseases: {0}'.format(', '.join(unused_diseases)))
    diseases = [d for d in all_diseases if d not in unused_diseases]
    quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters, categories=diseases, n_files=n_files)

    classifier_name = 'classifier_f_{0}_w_{1}'.format(n_files, '.'.join(unused_diseases))
    classifier = Classifier(model_weights_file_path=du.read_model_path(classifier_name),
                            trainable=False)

    lsg_name = 'lsg_f.{0}_c.{1}_w.{2}'.format(n_files, n_clusters, '.'.join(unused_diseases))
    lsg = LowShotGenerator(classifier.model, quadruplets_data, λ=λ, name=lsg_name)

    # callback = CloudCallback(True, config.slack_url, config.stop_url, config.slack_channel, name=lsg_name)
    # lsg.fit(callbacks=[callback])

    lsg.fit()

    unused_data = collect.load_quadruplets(n_clusters=n_clusters, categories=unused_diseases, n_files=n_files)
    quadruplets, centroids, cat_to_vectors, cat_to_onehots, original_shape = unused_data

    n_examples = min(len(ls) for ls in cat_to_vectors.values())

    n_samples = 100
    disease = 'Pneumonia'
    print('Generating {0} examples from {1} samples of {2}'.format(n_examples, n_samples, disease))

    X_train_disease, X_test_disease = cat_to_vectors[disease][:n_samples], cat_to_vectors[disease][n_samples:]
    X_train_disease = np.concatenate(
        [X_train_disease] + [lsg.generate(ϕ, n_new=n_examples // n_samples) for ϕ in X_train_disease])
    y_train_disease = np.array([cat_to_onehots[disease] for x in X_train_disease])
    y_test_disease = np.array([cat_to_onehots[disease] for x in X_test_disease])

    data_obj = du.get_processed_data(num_files_to_fetch_data_from=12)
    X, y = du.get_features_and_labels(data_obj)
    X, y = du.remove_diseases(X, y, unused_diseases, data_obj)
    X_train, X_test, y_train, y_test = du.get_train_test_split(X, y, test_size=0.1)
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

    args = parser.parse_args()

    main(args.n_clusters, args.n_files, args.λ, args.test)
