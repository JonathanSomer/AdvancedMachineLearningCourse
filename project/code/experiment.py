from sklearn.externals import joblib
from classifier import Classifier
from generator import LowShotGenerator
from sklearn.model_selection import train_test_split

import argparse
import data_utils as du


def log_fn(data):
    print(data)


def run(novel_disease_name, n_examples, n_samples_to_generate, λ, n_clusters, n_files):
    # Define the name of the experiment
    experiment_name = '{0}.{1}_examples.{2}_generated.{3}_lambda.{4}_clusters.{5}_files'.format(novel_disease_name,
                                                                                                n_examples,
                                                                                                n_samples_to_generate,
                                                                                                λ,
                                                                                                n_clusters,
                                                                                                n_files)

    # Initial Fetch Data
    data_obj = du.get_processed_data(num_files_to_fetch_data_from=n_files)
    le = du.get_label_encoder(data_obj)
    novel_disease_label = le.transform((novel_disease_name,))[0]

    X, y = du.get_features_and_labels(data_obj)

    X_train, X_test, y_train, y_test = du.get_train_test_split_without_disease(X, y, novel_disease_name, data_obj)

    # Train a classifier on all diseases but one
    cls = Classifier(n_classes=du.N_CLASSES-1, n_epochs=1)
    cls.fit(X_train, y_train)
    log_fn("accuracy acheived: %f" % cls.evaluate(X_test, y_test)[1])

    # Use the trained classifier and generate new examples
    split = du.get_train_test_split_with_n_samples_of_disease(X, y, novel_disease_name, data_obj, n_examples)
    X_train, X_test, y_train, y_test, n_samples_features, n_samples_integer_labels = split

    generated_features = LowShotGenerator.get_generated_features(cls,
                                                                 novel_disease_label,
                                                                 n_samples_features,
                                                                 n_clusters,
                                                                 λ,
                                                                 n_samples_to_generate)

    # Train a classifier on real n_samples of one disease
    split = du.get_train_test_split_with_n_samples_of_disease(X, y, novel_disease_name, data_obj, n_examples)
    X_train, X_test, y_train, y_test, n_samples_features, n_samples_integer_labels = split

    cls = Classifier(n_classes=du.N_CLASSES, n_epochs=1)
    cls.fit(X_train, y_train)
    log_fn("accuracy acheived: %f" % cls.evaluate(X_test, y_test)[1])

    # Train a classifier on real n_samples + generated samples
    generated_data_label = novel_disease_label
    X_train, X_test, y_train, y_test = du.get_train_test_with_generated_data(X_train, X_test, y_train, y_test,
                                                                             generated_features, generated_data_label)

    cls = Classifier(n_classes=du.N_CLASSES, n_epochs=1)
    cls.fit(X_train, y_train)
    log_fn("accuracy acheived: %f" % cls.evaluate(X_test, y_test)[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('disease', help='novel disease name')
    parser.add_argument('-g', '--n_samples', help='number of samples to generate', type=int, default=20)
    parser.add_argument('-e', '--n_examples', help='number of examples to use', type=int, default=1)
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-c', '--n_clusters', help='number of clusters to create', type=int, default=20)
    parser.add_argument('-l', '--lambda_value', help='value of lambda', type=float, default=.5)

    args = parser.parse_args()

    run(args.disease, args.n_examples, args.n_samples, args.lambda_value, args.n_clusters, args.n_files)
