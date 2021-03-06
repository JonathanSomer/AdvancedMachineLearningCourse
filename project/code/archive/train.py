from classifier import Classifier
from callbacks import CloudCallback
from sklearn.externals import joblib

import config
import data_utils as du
import argparse
import pandas as pd
import requests
import os


def update(msg):
    payload = {'message': msg, 'channel': config.slack_channel}
    requests.post(config.slack_url, json=payload)


def do_print(data):
    print(data)


# TODO: log results into files or into slack channel for results only.
def get_trained_classifier_and_data(diseases_to_remove, test_size=.1, n_files=12, logging_func=do_print,
                                    force_training=False):
    name = 'classifier_f_{0}_w_{1}'.format(n_files, '.'.join(diseases_to_remove))

    n_diseases = 15
    n_classes = n_diseases - len(diseases_to_remove)

    model_path = du.read_model_path(name)
    data_path = du.read_pickle_path(name)
    if os.path.exists(model_path) and os.path.exists(data_path) and not force_training:
        logging_func('Loaded classifier and data from files')
        classifier = Classifier(model_weights_file_path=model_path, n_classes=n_classes)
        X_train, X_test, y_train, y_test = joblib.load(data_path)
        return classifier, X_train, X_test, y_train, y_test

    logging_func('Fitting classifier without the diseases: *{0}*'.format(', '.join(diseases_to_remove)))
    data_obj = du.get_processed_data(num_files_to_fetch_data_from=n_files)

    # diseases = data_obj['label_encoder_classes']
    # n_classes = len(diseases) - len(diseases_to_remove)
    # n_classes = len(diseases)

    X, y = du.get_features_and_labels(data_obj)
    X_filtered, y_filtered = du.remove_diseases(X, y, diseases_to_remove, data_obj)
    X_train, X_test, y_train, y_test = du.get_train_test_split(X_filtered, y_filtered, test_size=test_size)

    joblib.dump((X_train, X_test, y_train, y_test), du.write_pickle_path(name))

    classifier = Classifier(n_classes=n_classes)
    classifier.fit(X_train,
                   y_train,
                   model_weights_file_path=du.write_model_path(name))

    logging_func('Evaluating with `{0}` of the data'.format(test_size))
    loss, acc = classifier.evaluate(X_test, y_test)
    logging_func('\naccuracy acheived: `{0}`'.format(acc))

    return classifier, X_train, X_test, y_train, y_test


def main(n_files, test_size, test, stop_instance):
    if test:
        n_files = 1

    update('*Training procedure has just started* :weight_lifter:')
    update('Fetching processed data from {0} {1}'.format(n_files, 'files' if n_files > 1 else 'file'))
    data_obj = du.get_processed_data(num_files_to_fetch_data_from=n_files)
    update('done :tada:')

    diseases = data_obj['label_encoder_classes']
    diseases_to_remove = diseases[12:]
    n_classes = len(diseases) - len(diseases_to_remove)
    # n_classes = len(diseases)

    update('Fitting classifier without the diseases: *{0}*'.format(', '.join(diseases_to_remove)))
    X, y = du.get_features_and_labels(data_obj)
    X_filtered, y_filtered = du.remove_diseases(X, y, diseases_to_remove, data_obj)
    X_train, X_test, y_train, y_test = du.get_train_test_split(X_filtered, y_filtered, test_size=test_size,
                                                               n_classes=n_classes)

    # TODO: maybe we can use int labels (not onehot) + sparse_categorical_crossentropy!
    classifier = Classifier(n_classes=n_classes)

    name = 'classifier_f_{0}_w_{1}'.format(n_files, '.'.join(diseases_to_remove))
    callbacks = [CloudCallback(True, config.slack_url, config.stop_url, config.slack_channel, name=name)]

    classifier.fit(X_train,
                   y_train,
                   model_weights_file_path=du.write_model_path(name),
                   callbacks=callbacks)
    update('done :tada: classifier saved as *{0}.h5*'.format(name))

    update('Evaluating with `{0}` of the data'.format(0.1))
    loss, acc = classifier.evaluate(X_test, y_test)
    update('done :tada: accuracy acheived: `{0}`'.format(acc))

    if stop_instance:
        update('Stopping instance')
        requests.get(config.stop_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-s', '--stop_instance', help='stop instance when run ends or not', action='store_true')
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')
    parser.add_argument('-ts', '--test_size', help='required test size for data splitting', type=int, default=.1)

    args = parser.parse_args()

    main(args.n_files, args.test_size, args.test, args.stop_instance)
