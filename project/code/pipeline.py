import os
import logger
import warnings
import matplotlib.pyplot as plt
from config import *
from sklearn.metrics import roc_curve, auc
from mnist_classifier import *
from mnist_data import *

_logger = logger.get_logger(__name__)


class PipeLine:
    def __init__(self, dataset, cls):
        self.dataset = dataset
        self.cls = cls
        self.base_results = self._base_results()

    def _base_results(self):
        _logger.info('get base results')
        self.cls.fit(*self.dataset.into_fit())

        results, fpr, tpr = self.evaluate_cls()
        self.create_cls_roc_plot(fpr, tpr, results, 'base line results')
        return results

    def evaluate_cls(self, removed_inx=None):
        results = {}

        y_score = self.cls.model.predict(self.dataset.x_test)
        fpr = dict()
        tpr = dict()
        n_classes = self.dataset.get_num_classes()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for inx in range(n_classes):
                results[inx] = {}

                loss, acc = self.cls.evaluate(*self.dataset.into_evaluate(inx))
                results[inx]['accuracy'] = acc

                fpr[inx], tpr[inx], _ = roc_curve(*self.dataset.into_roc_curve(y_score, inx))
                results[inx]['auc'] = auc(fpr[inx], tpr[inx])

        return results, fpr, tpr

    def create_cls_roc_plot(self, fpr, tpr, results, figure_name, sub_directory=None):
        plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')
        lw = 2
        fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')
        n_classes = self.dataset.get_num_classes()
        for inx in range(n_classes):
            plt.plot(fpr[inx], tpr[inx], lw=lw,
                     label='{0} (area = {1:0.5f})'.format(inx, results[inx]['auc']))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves %s' % figure_name)
        plt.legend(loc="lower right")

        if sub_directory:
            abs_path = os.path.join(local_results_dir, sub_directory)
        else:
            abs_path = local_results_dir

        fig.savefig('%s.png' % figure_name.replace(" ", "_"), dpi=fig.dpi)


if __name__ == "__main__":
    cls = MnistClassifier()
    d = MnistData(use_data_subset=True)
    PipeLine(d, cls)