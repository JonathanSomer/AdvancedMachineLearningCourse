import os
import logger
import warnings
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from config import *
from sklearn.metrics import roc_curve, auc
from mnist_classifier import *
from mnist_data import *
from generator import *

_logger = logger.get_logger(__name__)
N_GIVEN_EXAMPLES = [1, 2, 5, 10, 20, 40, 60, 80, 100]


class PipeLine:
    def __init__(self, dataset, cls):
        self.dataset = dataset
        self.cls = cls
        self.n_classes = self.dataset.get_num_classes()
        self.base_results = self._base_results()
        self.low_shot_results = {}
        for i in range(self.n_classes):
            self.low_shot_results[i] = self.get_low_shot_results(i)

    def _base_results(self):
        _logger.info('get base results')
        self.cls.fit(*self.dataset.into_fit())

        results, fpr, tpr = self.evaluate_cls()
        self.create_cls_roc_plot(fpr, tpr, results, 'base line results')
        return results

    def evaluate_cls(self, removed_inx=None):
        results = {}

        x_temp, y_one_hot = self.dataset.into_evaluate() #TODO -> RENAME!
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
        self.dataset.set_removed_class(inx)

        self.dataset.set_removed_class(class_index=inx, verbose=True)
        self.cls.fit(*self.dataset.into_fit())
        #results, fpr, tpr = self.evaluate_cls(removed_inx=inx)
        generator = LowShotGenerator(self.cls, self.dataset)

        for n_examples in N_GIVEN_EXAMPLES:
            low_shot_learning_results[n_examples] = {}

            self.dataset.set_number_of_samples_to_use(n=n_examples)

            self.cls.fit(*self.dataset.into_fit())

            results, fpr, tpr = self.evaluate_cls()
            #self.create_cls_roc_plot(fpr, tpr, results, '%d - with %d samples without generated data' % (inx,n_examples))
            low_shot_learning_results[n_examples]['without'] = results[inx]

            n_examples = self.dataset.x_class_removed_train[:self.n_examples]

            generated_data = generator.generate_from_samples(n_examples, n_total=N_GIVEN_EXAMPLES[-1],
                                                             smart_category=False, smart_centroids=False)
            self.dataset.set_generated_data(generated_data)
            self.cls.fit(*self.dataset.into_fit())

            results, fpr, tpr = self.evaluate_cls()
            #self.create_cls_roc_plot(fpr, tpr, results,
            #                         '%d - with %d samples without generated data' % (inx, n_examples))
            low_shot_learning_results[n_examples]['with no category'] = results[inx]

            generated_data = generator.generate_from_samples(n_examples, n_total=N_GIVEN_EXAMPLES[-1],
                                                             smart_category=True, smart_centroids=False)
            self.dataset.set_generated_data(generated_data)
            self.cls.fit(*self.dataset.into_fit())

            results, fpr, tpr = self.evaluate_cls()
            # self.create_cls_roc_plot(fpr, tpr, results,
            #                         '%d - with %d samples without generated data' % (inx, n_examples))
            low_shot_learning_results[n_examples]['with true category'] = results[inx]

            self.dataset.set_generated_data(None)

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
        auc_without = [low_shot_results[n]['without']['auc'] for n in N_GIVEN_EXAMPLES]
        accuracy_without = [low_shot_results[n]['without']['accuracy'] for n in N_GIVEN_EXAMPLES]

        auc_with_f = [low_shot_results[n]['with no category']['auc'] for n in N_GIVEN_EXAMPLES]
        accuracy_with_f = [low_shot_results[n]['with no category']['accuracy'] for n in N_GIVEN_EXAMPLES]

        auc_with_t = [low_shot_results[n]['with true category']['auc'] for n in N_GIVEN_EXAMPLES]
        accuracy_with_t = [low_shot_results[n]['with true category']['accuracy'] for n in N_GIVEN_EXAMPLES]


        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

        plt.plot(N_GIVEN_EXAMPLES, auc_without, marker='o', label='auc without generated data')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_without, marker='o', label='accuracy without generated data')

        plt.plot(N_GIVEN_EXAMPLES, auc_with_f, marker='o', label='auc with generated data not category')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_with_f, marker='o', label='accuracy with generated data not category')

        plt.plot(N_GIVEN_EXAMPLES, auc_with_t, marker='o', label='auc with generated data true category')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_with_t, marker='o', label='accuracy with generated data true category')

        plt.xlabel('number of examples')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title('%s - base results: auc %f accuracy %f' % (figure_name, base_results['auc'], base_results['accuracy']))

        figure_save_name = '%s.png' % figure_name.replace(" ", "_")
        fig.savefig(os.path.join(local_results_dir, figure_save_name), dpi=fig.dpi)

if __name__ == "__main__":
    cls = MnistClassifier()
    d = MnistData(use_data_subset=True)
    PipeLine(d, cls)
