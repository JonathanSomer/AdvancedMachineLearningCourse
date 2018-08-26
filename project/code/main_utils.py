import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from generator import LowShotGenerator
from data_utils import *
from classifier import *
from config import *
import itertools
import shutil

import logger
_logger = logger.get_logger(__name__)

N_GIVEN_EXAMPLES = [1,2,5,10,20]
NUMBER_OF_TOTAL_GENRATED_EXAMPLES = 20

NUMBER_OF_CLUSTERS = 20
λ = 0.5

ALL_DISEASES_MODEL = 'model_trained_on_all_diseases'
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

NUMBER_OF_DISEASES = len(all_diseases)


def get_raw_data(n_files):
    _logger.info('get raw results')
    return get_processed_data(num_files_to_fetch_data_from=n_files)


def get_base_results(raw_data):
    _logger.info('get base results')
    cls = Classifier(n_epochs=2) #TODO -> remove n_epochs for full run!
    X, y = get_features_and_labels(raw_data)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2)
    cls.fit(X_train, y_train, model_weights_file_path=read_model_path(ALL_DISEASES_MODEL))

    results = evaluate_cls(cls, X_test, y_test)
    create_cls_roc_plot(results, 'base line results')
    return results


def evaluate_cls(cls, X_test, y_test):
    results = {}

    y_score = cls.model.predict(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, disease in enumerate(all_diseases):
            _logger.info('\t %s' % disease)
            results[disease] = {}

            mask = (y_test[:, i] == 1.0)
            y_test_sub = y_test[mask]
            X_test_sub = X_test[mask]
            loss, acc = cls.evaluate(X_test_sub, y_test_sub)
            results[disease]['accuracy'] = acc

            results[disease]['fpr'], results[disease]['tpr'], _ = roc_curve(y_test[:, i], y_score[:, i])
            results[disease]['auc'] = auc(results[disease]['fpr'], results[disease]['tpr'])

    return results


def store_base_results(results, filename):
    pass


def extract_base_results(filename):
    return 8


def get_low_shot_results(raw_data):
    low_shot_learning_results = {}
    low_show_dir_path = 'low_shot'
    X, y = get_features_and_labels(raw_data)
    le = get_label_encoder(raw_data)

    for disease in all_diseases:
        disease_path = os.path.join('low_shot', disease)
        _logger.info('\t %s' % disease)
        low_shot_learning_results[disease] = {}


        novel_disease_label = le.transform((disease,))[0]
        X_train, X_test, y_train, y_test = get_train_test_split_without_disease(X, y, disease, raw_data)
        disease_base_cls = Classifier(n_classes=N_CLASSES - 1, n_epochs=1)
        disease_base_cls.fit(X_train, y_train)
        cls_results = evaluate_cls(disease_base_cls, X_evaluation, y_evaluation)
        create_cls_roc_plot(cls_results, '%s base cls' % disease, disease_path)

        for n_examples in N_GIVEN_EXAMPLES:
            temp = get_train_test_split_with_n_samples_of_disease(X, y, disease, raw_data, n_examples)
            X_train, X_test, y_train, y_test, n_samples_features, n_samples_integer_labels = temp
            generated_features = LowShotGenerator.get_generated_features(disease_base_cls,
                                                                         novel_disease_label,
                                                                         n_samples_features,
                                                                         NUMBER_OF_CLUSTERS,
                                                                         λ,
                                                                         NUMBER_OF_TOTAL_GENRATED_EXAMPLES)

            temp = get_train_test_split_with_n_samples_of_disease(X, y, disease, raw_data, n_examples)
            X_train, X_test, y_train, y_test, n_samples_features, n_samples_integer_labels = temp

            cls_with_generator = Classifier(n_classes=N_CLASSES, n_epochs=1)
            cls_with_generator.fit(X_train, y_train)

            temp = get_train_test_with_generated_data(X_train, X_test, y_train, y_test, generated_features, novel_disease_label)
            X_train, X_test, y_train, y_test = temp
            cls_without_generator = Classifier(n_classes=N_CLASSES, n_epochs=1)
            cls_without_generator.fit(X_train, y_train)

            results_with_generator = evaluate_cls(cls_with_generator, X_evaluation, y_evaluation)
            results_without_generator = evaluate_cls(cls_without_generator, X_evaluation, y_evaluation)
            create_cls_roc_plot(results_with_generator, '%d examples with generated samples'%n_examples, disease_path)
            create_cls_roc_plot(results_without_generator, '%d examples without generated samples'%n_examples, disease_path)

            low_shot_learning_results[disease][n_examples] = {'with':results_with_generator[disease],
                                                              'without':results_without_generator[disease]}

    return low_shot_learning_results


def store_low_shot_results(results, filename):
    pass


def extract_low_shot_results(filename):
    return None


def get_improved_low_shot_results(raw_data):
    return None


def store_improved_low_shot_results(results, filename):
    pass


def extract_improved_low_shot_results(filename):
    return None


def parse_results(results):
    _logger.info('parse results')
    for disease in all_diseases:
        _logger.info('result for %s' % disease)
        base_results = results['base results'][disease]
        _logger.info('base line accuracy %f, auc %f' % (base_results['accuracy'], base_results['auc']))

        for n_examples in N_GIVEN_EXAMPLES:
            _logger.info('low shot results:')
            low_shot_results_n_with = results['low shot results'][disease][n_examples]['with']
            low_shot_results_n_without = results['low shot results'][disease][n_examples]['without']
            _logger.info('%d samples with generator accuracy %f, auc %f' % (n_examples,
                                                                            low_shot_results_n_with['accuracy'],
                                                                            low_shot_results_n_with['auc']))
            _logger.info('%d samples without generator accuracy %f, auc %f' % (n_examples,
                                                                            low_shot_results_n_without['accuracy'],
                                                                            low_shot_results_n_without['auc']))

        path = os.path.join('low_shot', disease)
        #create_low_shot_results_plot(results['low shot results'][disease], '%s low shot results' % disease, path)
        _logger.info('\n')


def create_low_shot_results_plot_tester():
    result = {}
    for n in N_GIVEN_EXAMPLES:
        result[n] = {'with':{'auc':n*100, 'accuracy':n*200},'without':{'auc':n*300, 'accuracy':n*400}}

    create_low_shot_results_plot(result, 'example')


def create_cls_roc_plot(results, figure_name, sub_directory=None):
    plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')
    lw = 2
    fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')
    for i, disease in enumerate(results.keys()):
        plt.plot(results[disease]['fpr'], results[disease]['tpr'], lw=lw,
                 label='{0} (area = {1:0.2f})'.format(disease, results[disease]['auc']))

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

    fig.savefig(os.path.join(abs_path, '%s.png'%figure_name), dpi=fig.dpi)


def create_low_shot_results_plot(results, figure_name, sub_directory='low_shot'):
    auc_with = [results[n]['with']['auc'] for n in N_GIVEN_EXAMPLES]
    auc_without = [results[n]['without']['auc'] for n in N_GIVEN_EXAMPLES]
    accuracy_with = [results[n]['with']['accuracy'] for n in N_GIVEN_EXAMPLES]
    accuracy_without = [results[n]['without']['accuracy'] for n in N_GIVEN_EXAMPLES]

    fig = plt.figure(figsize=(12, 10), dpi=160, facecolor='w', edgecolor='k')

    plt.plot(N_GIVEN_EXAMPLES, auc_with, marker='o', label='auc with generated data')
    plt.plot(N_GIVEN_EXAMPLES, auc_without, marker='o', label='auc without generated data')
    plt.plot(N_GIVEN_EXAMPLES, accuracy_with, marker='o', label='accuracy with generated data')
    plt.plot(N_GIVEN_EXAMPLES, accuracy_without, marker='o', label='accuracy without generated data')

    plt.xlabel('number of examples')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title(figure_name)
    abs_path = os.path.join(local_results_dir, sub_directory)
    fig.savefig(os.path.join(abs_path, '%s.png' % figure_name), dpi=fig.dpi)

'''
def table():
    ab = itertools.chain(['acc', 'tau'] * (1+len(N_GIVEN_EXAMPLES)))
    collabel = list(ab)
    plt.figure(figsize=(20, 10), dpi=100, facecolor='w', edgecolor='k')

    header_0 = plt.table(cellText=[[''] * (len(N_GIVEN_EXAMPLES)+1)],
                         colWidths=[0.1] * (len(N_GIVEN_EXAMPLES)+1),
                         colLabels=['base results'] + ['%d'%n for n in N_GIVEN_EXAMPLES],
                         loc='bottom',
                         bbox=[0, -0.1, 0.8, 0.1]
                         )

    rowLabels = list(itertools.chain(*[['%s with' % disease, '%s without' % disease] for disease in all_diseases]))
    print(rowLabels)
    plt.axis('tight')
    plt.axis('off')
    plt.title('table!')
    clust_data = np.random.random((len(all_diseases)*2, 2 * (1 + len(N_GIVEN_EXAMPLES))))
    clust_data = [['%.4f' % j for j in i] for i in clust_data]
    plt.table(  cellText=clust_data,
                colWidths=[0.07 for x in collabel],
                colLabels=collabel,
                rowLabels=rowLabels,
                loc='center',
                bbox = [0, -0.35, 1.0, 0.9])

    plt.show()'''


def create_sub_directory():
    if os.path.exists(local_results_dir):
        shutil.rmtree(local_results_dir)

    os.mkdir(local_results_dir)
    subdirs = ['low_shot', 'enhanced_low_shot']
    for dir in subdirs:
        dir_path = os.path.join(local_results_dir, dir)
        os.mkdir(dir_path)
        for disease in all_diseases:
            disease_path = os.path.join(dir_path, disease)
            os.mkdir(disease_path)