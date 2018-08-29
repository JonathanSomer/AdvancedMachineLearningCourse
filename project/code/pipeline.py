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

        y_score = self.cls.model.predict(self.dataset.into_evaluate()[0])
        fpr = dict()
        tpr = dict()
        n_classes = self.dataset.get_num_classes()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for inx in range(n_classes):
                results[inx] = {}

                loss, acc = self.cls.evaluate(*self.dataset.into_evaluate_one_class(inx))
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

    def get_low_shot_results(self, inx):
        low_shot_learning_results = {}
        X, y = get_features_and_labels(raw_data)
        le = get_label_encoder(raw_data)

        for disease in queries_diseases:
            disease_path = os.path.join('low_shot', disease)
            _logger.info('\t %s' % disease)
            low_shot_learning_results[disease] = {}

            novel_disease_label = le.transform((disease,))[0]
            X_train, X_test, y_train, y_test = new_get_train_test_split_without_disease(X, y, disease, raw_data)
            disease_base_cls = Classifier(n_classes=N_CLASSES - 1, n_epochs=1)
            disease_base_cls.fit(X_train, new_onehot_encode(y_train, [novel_disease_label]))
            cls_results, fpr, tpr = evaluate_cls(disease_base_cls, X_test, y_test,
                                                 diseases_removed=[novel_disease_label])
            create_cls_roc_plot(fpr, tpr, cls_results, '%s base cls' % disease, disease_path)

            temp = get_all_disease_samples_and_rest(X, y, novel_disease_label, raw_data)
            all_samples_features, all_samples_labels, rest_features, rest_labels = temp
            X_test_with_disease, y_test_with_disease = add_disease_to_test_data(X_test, y_test, rest_features,
                                                                                rest_labels)
            for n_examples in N_GIVEN_EXAMPLES:
                temp = add_n_samples_to_train_data(X_train, y_train, all_samples_features, all_samples_labels,
                                                   n_examples)
                X_train_with_disease_samples, y_train_with_disease_samples, n_samples_features = temp

                cls_without_generator = Classifier(n_classes=N_CLASSES, n_epochs=1)
                cls_without_generator.fit(X_train_with_disease_samples, new_onehot_encode(y_train_with_disease_samples))
                results_without_generator, fpr_without, tpr_without = evaluate_cls(cls_without_generator,
                                                                                   X_test_with_disease,
                                                                                   y_test_with_disease)
                plt_name = '%d examples without generated samples' % n_examples
                create_cls_roc_plot(fpr_without, tpr_without, results_without_generator, plt_name, disease_path)

                generated_features = LowShotGenerator.get_generated_features(disease_base_cls,
                                                                             novel_disease_label,
                                                                             n_samples_features,
                                                                             NUMBER_OF_CLUSTERS,
                                                                             λ,
                                                                             NUMBER_OF_TOTAL_GENRATED_EXAMPLES)

                temp = add_generated_data_to_train_data(X_train_with_disease_samples, y_train_with_disease_samples,
                                                        generated_features, novel_disease_label)
                X_train_with_generated_data, y_train_with_generated_data = temp

                cls_with_generator = Classifier(n_classes=N_CLASSES, n_epochs=1)
                cls_with_generator.fit(X_train_with_generated_data, new_onehot_encode(y_train_with_generated_data))

                results_with_generator, fpr_with, tpr_with = evaluate_cls(cls_with_generator, X_test_with_disease,
                                                                          y_test_with_disease)
                create_cls_roc_plot(fpr_with, tpr_with, results_with_generator,
                                    '%d examples with generated samples' % n_examples, disease_path)

                low_shot_learning_results[disease][n_examples] = {'with': results_with_generator[disease],
                                                                  'without': results_without_generator[disease]}

        return low_shot_learning_results

if __name__ == "__main__":
    cls = MnistClassifier()
    d = MnistData(use_data_subset=True)
    PipeLine(d, cls)
