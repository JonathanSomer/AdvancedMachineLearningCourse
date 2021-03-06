import pipeline
import argparse
import config
import logger
import os
import shutil
import glob

_logger = logger.get_logger(__name__)
hidden_sizes = (64, 128, 256,)
λs = (.25, .5, .75,)


def run_cv(dataset_type, cls_type, use_data_subset=False, use_features=True, use_class_weights=True,
                 generator_epochs=2, classifier_epochs=12, n_clusters=30, fix_class_imbalance=False):

    base_dir = config.local_results_dir
    for hidden_size in hidden_sizes:
        for λ in λs:
            _logger.info('run for  hidden_size=%d and λ=%f' % (hidden_size, λ))
            pipeline.Pipeline(dataset_type, cls_type, use_data_subset, use_features, use_class_weights,
                 generator_epochs, classifier_epochs, n_clusters, hidden_size, λ, fix_class_imbalance=fix_class_imbalance)

            new_dir_path = os.path.join(base_dir, '%d_%f'%(hidden_size, λ))
            os.makedirs(new_dir_path)
            for png_file in glob.glob(os.path.join(base_dir, '*.png')):
                shutil.copy2(png_file, new_dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='what dataset to use', default='mnist')
    parser.add_argument('-t', '--test', help='is it a test run or not', action='store_true')
    parser.add_argument('-ce', '--classifier_epochs', help='number of epochs to the train classifier with', type=int,
                        default=12)
    parser.add_argument('-ge', '--generator_epochs', help='number of epochs to train the generator with', type=int,
                        default=2)
    parser.add_argument('-r', '--raw_data', help='whether to use raw data (and not features) or not',
                        action='store_true')
    parser.add_argument('-ww', '--without_weights', help='whether to disable class_weights or not',
                        action='store_true')
    parser.add_argument('-c', '--n_clusters', help='number of clusters to use', type=int, default=30)
    parser.add_argument('-n', '--n_total', help='number of examples + generated', type=int, default=pipeline.N_GIVEN_EXAMPLES[-1])
    parser.add_argument('-fc', '--fix_class_imbalance', help='uniformally sample from classes', action='store_true')

    args = parser.parse_args()

    if args.dataset in pipeline.DATA_SETS and args.dataset in pipeline.CLSES:
        run_cv( pipeline.DATA_SETS[args.dataset],
                pipeline.CLSES[args.dataset],
                use_data_subset=args.test,
                use_features=not args.raw_data,
                use_class_weights=not args.without_weights,
                classifier_epochs=args.classifier_epochs,
                generator_epochs=args.generator_epochs,
                n_clusters=args.n_clusters,
                fix_class_imbalance=args.fix_class_imbalance)
    else:
        _logger.error('unknown dataset %s' % args.dataset)
