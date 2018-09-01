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
from cifar_classifier import *
from cifar_data import *
from generator import *
import functools

_logger = logger.get_logger(__name__)
N_GIVEN_EXAMPLES = [1, 2, 5, 10, 20]
N_TOTALS = [20, 50, 100, 300]


class Pipeline(object):
    def __init__(self, dataset_type, cls_type, use_data_subset=False, use_features=True, use_class_weights=True,
                 generator_epochs=2, classifier_epochs=12, n_clusters=30, n_total=N_GIVEN_EXAMPLES[-1],
                 λ=.95, hidden_size=256, fix_class_imbalance = False):

        self.dataset_type = dataset_type
        self.use_data_subset = use_data_subset
        self.use_features = use_features
        self.use_class_weights = use_class_weights
        self.generator_epochs = generator_epochs
        self.classifier_epochs = classifier_epochs
        self.n_clusters = n_clusters
        self.n_total = n_total
        self.cls_type = cls_type
        self.λ = λ
        self.hidden_size = hidden_size
        self.fix_class_imbalance = fix_class_imbalance

        self.dataset = dataset_type(use_features=self.use_features, use_data_subset=use_data_subset)
        self.cls = cls_type(use_features=self.use_features, epochs=self.classifier_epochs)

        self.n_classes = self.dataset.get_num_classes()
        self.base_results = {}#self._base_results()
        self.low_shot_results = {}
        for i in range(4):
            self.low_shot_results[i] = self.low_shot_test(i)
            self.generated_test(i)

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

    def get_generator(self, inx):
        _logger.info('create generator')
        temp_data_object = self.dataset_type(use_features=self.use_features, class_removed=inx)  # TODO -> refactor
        temp_data_object.set_removed_class(class_index=inx, verbose=True)
        self.dataset.set_removed_class(class_index=inx, verbose=True)

        all_but_one_classifier = self.cls_type(use_features=self.use_features, epochs=self.classifier_epochs)
        all_but_one_classifier.fit(*temp_data_object.into_fit())
        all_but_one_classifier.set_trainability(is_trainable=False)


        generator = LowShotGenerator(all_but_one_classifier.model,
                                     temp_data_object,
                                     epochs=self.generator_epochs,
                                     n_clusters=self.n_clusters,
                                     λ=self.λ,
                                     hidden_size=self.hidden_size)
        return generator

    def low_shot_test(self, inx):
        _logger.info('run low shot test for inx %d' % inx)
        results = {}
        generator = self.get_generator(inx)

        for n in N_GIVEN_EXAMPLES:
            _logger.info('number of examples is %d' % n)
            results[n] = {}
            self.dataset.set_number_of_samples_to_use(n=n)
            examples = self.dataset.get_n_samples(n)

            base_gen_func = functools.partial(generator.generate_from_samples, examples, n_total=self.n_total)
            generators_options = {'baseline': lambda: None,
                                  'baseline + gen': functools.partial(base_gen_func, smart_category=False, smart_centroids=False),
                                  'smart category': functools.partial(base_gen_func, smart_category=True, smart_centroids=False),
                                  #'duplicated gen': functools.partial(self.duplicated_generator, examples, n_total=self.n_total),
                                    }

            for option in generators_options.keys():
                if n != N_GIVEN_EXAMPLES[-1]:
                    generated_data = generators_options[option]()
                    if generated_data is not None:
                        self.dataset.set_generated_data(generated_data)

                self.cls.fit(*self.dataset.into_fit(self.fix_class_imbalance), use_class_weights=self.use_class_weights)
                temp_results, fpr, tpr = self.evaluate_cls()
                results[n][option] = temp_results[inx]
                results[n][option]['avarage_rest'] = np.mean([temp_results[x]['accuracy'] for x in temp_results.keys() if x != inx])

            self.dataset.set_generated_data(None)

        _logger.info('export results for %d' % inx)
        self.export_one_shot_learning_result(results, inx)

    def generated_test(self, inx):
        results = {}
        _logger.info('run low shot test for inx %d' % inx)
        generator = self.get_generator(inx)
        fixed_number_of_examples = 5
        _logger.info('number of examples is %d' % fixed_number_of_examples)

        for n_total in N_TOTALS:
            results[n_total] = {}
            self.dataset.set_number_of_samples_to_use(n=fixed_number_of_examples)
            examples = self.dataset.get_n_samples(fixed_number_of_examples)

            base_gen_func = functools.partial(generator.generate_from_samples, examples, n_total=self.n_total)
            generators_options = {'baseline + gen': functools.partial(base_gen_func, smart_category=False, smart_centroids=False),
                                  'smart category': functools.partial(base_gen_func, smart_category=True, smart_centroids=False)}
                                  #'duplicated gen': functools.partial(self.duplicated_generator, examples, n_total=self.n_total),
                                  # 'baseline': lambda: None,  # NOT A BUG - SANITY CHECK!}

            for option in generators_options.keys():
                if n_total > fixed_number_of_examples:
                    generated_data = generators_options[option]()
                    if generated_data is not None:
                        self.dataset.set_generated_data(generated_data)

                self.cls.fit(*self.dataset.into_fit(self.fix_class_imbalance), use_class_weights=self.use_class_weights)
                temp_results, fpr, tpr = self.evaluate_cls()
                results[n_total][option] = temp_results[inx]

            self.dataset.set_generated_data(None)

        _logger.info('export results for %d' % inx)
        self.export_generated_results(results, inx)

    def duplicated_generator(self, examples, n_total):
        num_examples_to_create = n_total - len(examples)
        if num_examples_to_create <= 0:
            return None
        copied_examples = np.array(examples, copy=True)

        while len(copied_examples) < num_examples_to_create:
            copied_examples = np.concatenate((copied_examples, examples))

        return copied_examples[:num_examples_to_create]

    def export_one_shot_learning_result(self, results, inx):
        _logger.info('export results for %d' % inx)
        #base_inx_results = self.base_results[inx]
        #_logger.info('base line accuracy %f, auc %f' % (base_inx_results['accuracy'], base_inx_results['auc']))
        generating_options = results[N_GIVEN_EXAMPLES[0]].keys()
        for n_examples in N_GIVEN_EXAMPLES:
            _logger.info('low shot results %d examples:' % n_examples)
            for option in generating_options:
                results_n_option = results[n_examples][option]
                _logger.info('\t%s accuracy %f, auc %f' % (option,
                                                         results_n_option['accuracy'],
                                                         results_n_option['auc']))

        self.create_low_shot_results_plot(results, inx)
        self.create_low_shot_sanity_plot(results, inx)
        _logger.info('\n')

    def export_generated_results(self, results, inx):
        _logger.info('export generated results for %d' % inx)
        #base_inx_results = self.base_results[inx]
        #_logger.info('base line accuracy %f, auc %f' % (base_inx_results['accuracy'], base_inx_results['auc']))
        generating_options = results[N_TOTALS[0]].keys()
        for n_total in N_TOTALS:
            _logger.info('low shot results with %d total examples:' % n_total)
            for option in generating_options:
                results_n_option = results[n_total][option]
                _logger.info('\t%s accuracy %f, auc %f' % (option,
                                                         results_n_option['accuracy'],
                                                         results_n_option['auc']))

        self.create_generated_results_plot(results, inx)
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

    def create_low_shot_results_plot(self, low_shot_results, inx):
        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

        generating_options = low_shot_results[N_GIVEN_EXAMPLES[0]].keys()

        for option in generating_options:
            accuracy_plot = [low_shot_results[n][option]['accuracy'] for n in N_GIVEN_EXAMPLES]
            plt.plot(N_GIVEN_EXAMPLES, accuracy_plot, marker='o', label=option)

        plt.xlabel('Real examples')
        plt.xticks(N_GIVEN_EXAMPLES)
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid()
        plt.title('low shot on label index')# %d - base results accuracy %f' % (inx, self.base_results[inx]['accuracy']))

        figure_save_name = 'low_shot_%d.png' % inx
        fig.savefig(os.path.join(local_results_dir, figure_save_name), dpi=fig.dpi)

    def create_low_shot_sanity_plot(self, low_shot_results, inx):
        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

        generating_options = low_shot_results[N_GIVEN_EXAMPLES[0]].keys()

        for option in generating_options:
            accuracy_plot = [low_shot_results[n][option]['avarage_rest'] for n in N_GIVEN_EXAMPLES]
            plt.plot(N_GIVEN_EXAMPLES, accuracy_plot, marker='o', label=option)

        plt.xlabel('Real examples')
        plt.xticks(N_GIVEN_EXAMPLES)
        plt.ylabel('Avarage accuracy on all classes but %d' % inx)
        plt.legend()
        plt.grid()
        plt.title('Sanity check for index %d' % inx)

        figure_save_name = 'sanity_%d.png' % inx
        fig.savefig(os.path.join(local_results_dir, figure_save_name), dpi=fig.dpi)

    def create_generated_results_plot(self, results, inx):
        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

        generating_options = results[N_TOTALS[0]].keys()

        for option in generating_options:
            accuracy_plot = [results[n_total][option]['accuracy'] for n_total in N_TOTALS]
            plt.plot(N_TOTALS, accuracy_plot, marker='o', label=option)

        plt.xlabel('generated examples')
        plt.xticks(N_TOTALS)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.title('number of total examples vs accuracy check for index %d' % inx)

        figure_save_name = 'generated_test_%d.png' % inx
        fig.savefig(os.path.join(local_results_dir, figure_save_name), dpi=fig.dpi)

DATA_SETS = {'mnist': MnistData,
             'cifar10': Cifar10Data}

CLSES = {'mnist': MnistClassifier,
         'cifar10': Cifar10Classifier}

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
    parser.add_argument('-n', '--n_total', help='number of examples + generated', type=int, default=N_GIVEN_EXAMPLES[-1])
    parser.add_argument('-fc', '--fix_class_imbalance', help='uniformally sample from classes', action='store_true')

    args = parser.parse_args()

    if args.dataset in DATA_SETS and args.dataset in CLSES:
        Pipeline(DATA_SETS[args.dataset],
                 CLSES[args.dataset],
                 use_data_subset=args.test,
                 use_features=not args.raw_data,
                 use_class_weights=not args.without_weights,
                 classifier_epochs=args.classifier_epochs,
                 generator_epochs=args.generator_epochs,
                 n_clusters=args.n_clusters,
                 n_total=args.n_total,
                 fix_class_imbalance = args.fix_class_imbalance)
    else:
        _logger.error('unknown dataset %s' % args.dataset)
