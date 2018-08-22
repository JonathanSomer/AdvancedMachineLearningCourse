from generator import LowShotGenerator
from classifier import Classifier
from callbacks import CloudCallback

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

    unused_diseases = ['Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    print('Unused diseases: {0}'.format(', '.join(unused_diseases)))
    diseases = [d for d in all_diseases if d not in unused_diseases]
    quadruplets_data = collect.load_quadruplets(n_clusters=n_clusters, categories=diseases, n_files=n_files)

    classifier_name = 'classifier_f_{0}_w_{1}'.format(n_files, '.'.join(unused_diseases))
    classifier = Classifier(model_weights_file_path=du.read_model_path(classifier_name),
                            trainable=False)

    lsg_name = 'lsg_f.{0}_c.{1}_w.{2}'.format(n_files, n_clusters, '.'.join(unused_diseases))
    lsg = LowShotGenerator(classifier, quadruplets_data, λ=λ, name=lsg_name)

    callbacks = [CloudCallback(True, config.slack_url, config.stop_url, config.slack_channel, name=lsg_name)]
    lsg.fit(callbacks=callbacks)

    unused_data = collect.load_quadruplets(n_clusters=n_clusters, categories=unused_diseases, n_files=n_files)
    quadruplets, centroids, cat_to_vectors, cat_to_onehots, original_shape = unused_data

    ϕ = cat_to_vectors['Pneumonia'][15]
    example = lsg.generate(ϕ, n_new=1)

    print(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=12)
    parser.add_argument('-c', '--n_clusters', help='number of clusters', type=int, default=20)
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')
    parser.add_argument('-l', '--λ', help='set λ regularization parameter', type=float, default=.5)

    args = parser.parse_args()

    main(args.n_clusters, args.n_files, args.λ, args.test)