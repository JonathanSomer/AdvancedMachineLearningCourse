import os
import logger
import warnings
import matplotlib.pyplot as plt
from config import *
from sklearn.metrics import roc_curve, auc
from mnist_classifier import *
from mnist_data import *

_logger = logger.get_logger(__name__)
N_GIVEN_EXAMPLES = [1,2,5,10,20]


class PipeLine:
    def __init__(self, dataset, cls):
        self.dataset = dataset
        self.cls = cls
        self.n_classes = self.dataset.get_num_classes()
        self.base_results = self._base_results()
        self.low_shot_results = {}
        for i in range(2):
            self.low_shot_results[i] = self.get_low_shot_results(i)

        self.parse_results()

    def _base_results(self):
        _logger.info('get base results')
        self.cls.fit(*self.dataset.into_fit())

        results, fpr, tpr = self.evaluate_cls()
        self.create_cls_roc_plot(fpr, tpr, results, 'base line results')
        return results

    def evaluate_cls(self, removed_inx=None):
        results = {}

        y_score = self.cls.predict(self.dataset.into_evaluate()[0])
        fpr = dict()
        tpr = dict()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for inx in range(self.n_classes):
                if inx == removed_inx:
                    continue
                results[inx] = {}

                loss, acc = self.cls.evaluate(*self.dataset.into_evaluate_one_class(inx))
                results[inx]['accuracy'] = acc

                fpr[inx], tpr[inx], _ = roc_curve(*self.dataset.into_roc_curve(y_score, inx))
                results[inx]['auc'] = auc(fpr[inx], tpr[inx])

        return results, fpr, tpr

    def get_low_shot_results(self, inx):
        low_shot_learning_results = {}
        self.dataset.set_removed_class(inx)

        self.dataset.set_removed_class(class_index=inx, verbose=True)
        cls.fit(*d.into_fit())
        results, fpr, tpr = self.evaluate_cls(removed_inx=inx)
        self.create_cls_roc_plot(fpr, tpr, results, '%d - base line' % inx)

        for n_examples in N_GIVEN_EXAMPLES:
            low_shot_learning_results[n_examples] = {}
            #d.set_number_of_samples_to_use(n=n_examples)
            #cls.fit(*d.into_fit())
            #results, fpr, tpr = self.evaluate_cls(removed_inx=inx)
            #self.create_cls_roc_plot(fpr, tpr, results, '%d - with %d samples without generator' % (inx,n_examples))
            #low_shot_learning_results[n_examples]['without'] = results[inx]
            low_shot_learning_results[n_examples]['without'] = {'auc': 0.3, 'accuracy': 0.6}

            generated_data = d.get_generated_data_stub()
            low_shot_learning_results[n_examples]['with'] = {'auc': 0.8, 'accuracy':0.1}

        return low_shot_learning_results

    def parse_results(self):
        _logger.info('parse results')
        for inx in range(2):
            _logger.info('result for class %d' % inx)
            base_inx_results = self.base_results[inx]
            _logger.info('base line accuracy %f, auc %f' % (base_inx_results['accuracy'], base_inx_results['auc']))

            current_low_shot = self.low_shot_results[inx]
            for n_examples in N_GIVEN_EXAMPLES:
                _logger.info('low shot results:')

                low_shot_results_n_with = current_low_shot[n_examples]['with']
                low_shot_results_n_without = current_low_shot[n_examples]['without']
                _logger.info('%d samples with generator accuracy %f, auc %f' % (n_examples,
                                                                                low_shot_results_n_with['accuracy'],
                                                                                low_shot_results_n_with['auc']))
                _logger.info('%d samples without generator accuracy %f, auc %f' % (n_examples,
                                                                                   low_shot_results_n_without[
                                                                                       'accuracy'],
                                                                                   low_shot_results_n_without['auc']))

            self.create_low_shot_results_plot(base_inx_results, current_low_shot, '%d low shot results' % inx)
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
        auc_with = [low_shot_results[n]['with']['auc'] for n in N_GIVEN_EXAMPLES]
        auc_without = [low_shot_results[n]['without']['auc'] for n in N_GIVEN_EXAMPLES]
        accuracy_with = [low_shot_results[n]['with']['accuracy'] for n in N_GIVEN_EXAMPLES]
        accuracy_without = [low_shot_results[n]['without']['accuracy'] for n in N_GIVEN_EXAMPLES]

        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

        plt.plot(N_GIVEN_EXAMPLES, auc_with, marker='o', label='auc with generated data')
        plt.plot(N_GIVEN_EXAMPLES, auc_without, marker='o', label='auc without generated data')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_with, marker='o', label='accuracy with generated data')
        plt.plot(N_GIVEN_EXAMPLES, accuracy_without, marker='o', label='accuracy without generated data')

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
