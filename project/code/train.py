from classifier import Classifier
from callbacks import CloudCallback

import config
import data_utils as du
import argparse
import pandas as pd
import requests


def update(msg):
    payload = {'message': msg, 'channel': config.slack_channel}
    requests.post(config.slack_url, json=payload)


def do_print(data):
    print(data)


# TODO: make save optional
# TODO: log results into files or into slack channel for results only.
# TODO: try to work with sparse?
def do_train(diseases_to_remove, test_size=.1, shallow_no_finding=False, n_files=12, logging_func=do_print):
    logging_func('Fitting classifier without the diseases: *{0}*'.format(', '.join(diseases_to_remove)))
    # if with_no_finding:
    #     logging_func('No Finding is excluded as well')
    #     diseases_to_remove += ['No Finding']

    data_obj = du.get_processed_data(num_files_to_fetch_data_from=n_files)

    diseases = data_obj['label_encoder_classes']
    # n_classes = len(diseases) - len(diseases_to_remove)
    n_classes = len(diseases)

    X, y = du.get_features_and_labels(data_obj)
    if shallow_no_finding:
        X, y = du.shallow_no_finding(X, y, data_obj)

    X_filtered, y_filtered = du.remove_diseases(X, y, diseases_to_remove, data_obj)
    X_train, X_test, y_train, y_test = du.get_train_test_split(X_filtered, y_filtered, test_size=test_size,
                                                               n_classes=n_classes)
    logging_func('working with total of {0} samples. Test size is {1}'.format(len(X_filtered), test_size))

    classifier = Classifier(n_classes=n_classes)
    name = 'classifier_f_{0}_w_{1}'.format(n_files, '.'.join(diseases_to_remove))
    classifier.fit(X_train,
                   y_train)
                   # model_weights_file_path=du.write_model_path(name))
    # logging_func('done :tada: classifier saved as *{0}.h5*'.format(name))

    logging_func('Evaluating with `{0}` of the data'.format(0.1))
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
    X_train, X_test, y_train, y_test = du.get_train_test_split(X_filtered, y_filtered, test_size=test_size, n_classes=n_classes)

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