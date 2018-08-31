import os
import logger
import warnings
import matplotlib as mpl
import argparse

mpl.use('Agg')
import matplotlib.pyplot as plt

from config import *
from sklearn.metrics import roc_curve, auc
from mnist_classifier import *
from mnist_data import *
from generator import *

_logger = logger.get_logger(__name__)
N_GIVEN_EXAMPLES = [1, 2, 5, 10, 20]


class Pipeline(object):
    def __init__(self, dataset_type, cls_type, use_data_subset=False, use_features=True, use_class_weights=True,
                 generator_epochs=2, classifier_epochs=12, n_clusters=30):
        self.dataset_type = dataset_type
        self.use_data_subset = use_data_subset
        self.use_features = use_features
        self.use_class_weights = use_class_weights
        self.generator_epochs = generator_epochs
        self.classifier_epochs = classifier_epochs
        self.n_clusters = n_clusters
        self.cls_type = cls_type

        self.dataset = dataset_type(use_features=self.use_features, use_data_subset=use_data_subset)
        self.cls = cls_type(use_features=self.use_features, epochs=self.classifier_epochs)

        self.n_classes = self.dataset.get_num_classes()
        self.base_results = self._base_results()
        self.low_shot_results = {}
        for i in range(self.n_classes):
            self.low_shot_results[i] = self.get_low_shot_results(i)

    def _base_results(self):
        _logger.info('get base results')
        self.cls.fit(*self.dataset.into_fit(), use_class_weights=self.use_class_weights)

        results, fpr, tpr = self.evaluate_cls()
        self.create_cls_roc_plot(fpr, tpr, results, 'base line results')
        return results

    def evaluate_cls(self, removed_inx=None):
        results = {}

        x_temp, y_one_hot = self.dataset.into_evaluate()
        y_score = self.cls.predict(x_temp)
        fpr = dict()
        tpr = dict()
        off = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for inx in range(self.n_classes):
                if inx == removed_inx:
                    off = -1
                    continue
                results[inx] = {}

                loss, acc = self.cls.evaluate(*self.dataset.into_evaluate_one_class(inx))
                results[inx]['accuracy'] = acc

                fpr[inx], tpr[inx], _ = roc_curve(y_one_hot[:, inx + off], y_score[:, inx + off])
                results[inx]['auc'] = auc(fpr[inx], tpr[inx])

        return results, fpr, tpr

    def get_low_shot_results(self, inx):
        low_shot_learning_results = {}

        _logger.info('fit a classifier without label %d' % inx)
        temp_data_object = self.dataset_type(use_features=self.use_features, class_removed=inx)  # TODO -> refactor
        temp_data_object.set_removed_class(class_index=inx, verbose=True)
        self.dataset.set_removed_class(class_index=inx, verbose=True)

        all_but_one_classifier = self.cls_type(use_features=self.use_features, epochs=self.classifier_epochs)
        all_but_one_classifier.fit(*temp_data_object.into_fit())
        all_but_one_classifier.set_trainability(is_trainable=False)

        _logger.info('create generator')
        generator = LowShotGenerator(all_but_one_classifier.model,
                                     temp_data_object,
                                     epochs=self.generator_epochs,
                                     n_clusters=self.n_clusters)

        for n in N_GIVEN_EXAMPLES:
            _logger.info('number of examples is %d' % n)
            low_shot_learning_results[n] = {}

            self.dataset.set_number_of_samples_to_use(n=n)

            self.cls.fit(*self.dataset.into_fit(), use_class_weights=self.use_class_weights)

            results, fpr, tpr = self.evaluate_cls()
            low_shot_learning_results[n]['without'] = results[inx]

            n_examples = self.dataset.x_class_removed_train[:n]
            if n != N_GIVEN_EXAMPLES[-1]:
                generated_data = generator.generate_from_samples(n_examples, n_total=N_GIVEN_EXAMPLES[-1],
                                                                 smart_category=False, smart_centroids=False)
                self.dataset.set_generated_data(generated_data)
            self.cls.fit(*self.dataset.into_fit(), use_class_weights=self.use_class_weights)

            results, fpr, tpr = self.evaluate_cls()
            low_shot_learning_results[n]['with no category'] = results[inx]

            if n != N_GIVEN_EXAMPLES[-1]:
                generated_data = generator.generate_from_samples(n_examples, n_total=N_GIVEN_EXAMPLES[-1],
                                                                 smart_category=True, smart_centroids=False)
                self.dataset.set_generated_data(generated_data)

            self.cls.fit(*self.dataset.into_fit(), use_class_weights=self.use_class_weights)

            results, fpr, tpr = self.evaluate_cls()
            low_shot_learning_results[n]['with true category'] = results[inx]

            self.dataset.set_generated_data(None)

        _logger.info('export results for %d' % inx)
        self.export_one_shot_learning_result(low_shot_learning_results, inx)

    def export_one_shot_learning_result(self, results, inx):
        _logger.info('export results for %d' % inx)
        base_inx_results = self.base_results[inx]
        _logger.info('base line accuracy %f, auc %f' % (base_inx_results['accuracy'], base_inx_results['auc']))

        for n_examples in N_GIVEN_EXAMPLES:
            _logger.info('low shot results:')

            results_n_without = results[n_examples]['without']
            results_n_with_f = results[n_examples]['with no category']
            results_n_with_t = results[n_examples]['with true category']

            _logger.info('%d samples without generator accuracy %f, auc %f' % (n_examples,
                                                                               results_n_without['accuracy'],
                                                                               results_n_without['auc']))
            _logger.info('%d samples with generator no category accuracy %f, auc %f' % (n_examples,
                                                                                        results_n_with_f['accuracy'],
                                                                                        results_n_with_f['auc']))

            _logger.info('%d samples with generator true category accuracy %f, auc %f' % (n_examples,
                                                                                          results_n_with_t['accuracy'],
                                                                                          results_n_with_t['auc']))

        self.create_low_shot_results_plot(base_inx_results, results, '%d low shot results' % inx)
        _logger.info('\n')

    def create_cls_roc_plot(self, fpr, tpr, results, figure_name):
        plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')
        lw = 2
        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')
        for inx in range(self.n_classes):
            if inx not in fpr:
                continue
            plt.plot(fpr[inx], tpr[inx], lw=lw,
                     label='{0} (area = {1:0.5f})'.format(inx, results[inx]['auc']))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves %s' % figure_name)
        plt.legend(loc="lower right")

        figure_save_name = '%s.png' % figure_name.replace(" ", "_")

        fig.savefig(os.path.join(local_results_dir, figure_save_name), dpi=fig.dpi)

    def create_low_shot_results_plot(self, base_results, low_shot_results, figure_name):
        accuracy_without = [low_shot_results[n]['without']['accuracy'] for n in N_GIVEN_EXAMPLES]
        accuracy_with_f = [low_shot_results[n]['with no category']['accuracy'] for n in N_GIVEN_EXAMPLES]
        accuracy_with_t = [low_shot_results[n]['with true category']['accuracy'] for n in N_GIVEN_EXAMPLES]

        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

        plt.plot(N_GIVEN_EXAMPLES, accuracy_without, marker='o', label='baseline')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_with_f, marker='o', label='baseline + gen')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_with_t, marker='o', label='smart-category')

        plt.xlabel('number of examples')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(
            '%s - base results: auc %f accuracy %f' % (figure_name, base_results['auc'], base_results['accuracy']))

        figure_save_name = '%s.png' % figure_name.replace(" ", "_")
        fig.savefig(os.path.join(local_results_dir, figure_save_name), dpi=fig.dpi)


DATA_SETS = {'mnist': MnistData}
CLSES = {'mnist': MnistClassifier}

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

    args = parser.parse_args()

    if args.dataset in DATA_SETS and args.dataset in CLSES:
        Pipeline(DATA_SETS[args.dataset],
                 CLSES[args.dataset],
                 use_data_subset=args.test,
                 use_features=not args.raw_data,
                 use_class_weights=not args.without_weights,
                 classifier_epochs=args.classifier_epochs,
                 generator_epochs=args.generator_epochs,
                 n_clusters=args.n_clusters)
    else:
        _logger.error('unknown dataset %s' % args.dataset)
