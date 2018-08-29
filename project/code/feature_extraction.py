from mnist_data import *
from mnist_classifier import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.externals import joblib
from data_utils import *
import argparse

MNIST = 'mnist'
CIFAR10 = 'cifar10'
XRAY = 'xray'

def main(dataset_name, use_data_subset):
	
	d, cls = get_data_obj_and_classifier(dataset_name=dataset_name, 
										 use_data_subset=use_data_subset)
	# generate using all classes
	generate_features(d, cls, dataset_name, class_removed=None)

	# generate ommiting each class in turn
	n_classes = d.get_num_classes()
	for class_index in range(n_classes):
		generate_features(d, cls, dataset_name, class_removed=class_index)		


def write_features_from_all_classes_path(dataset):
	return write_pickle_path(dataset + '_features')

def write_features_from_all_classes_but_one_path(dataset, class_removed):
	return write_pickle_path(dataset + '_features' + '_no_' + str(class_removed))

def read_features_from_all_classes_path(dataset):
	return read_pickle_path(dataset + '_features')

def read_features_from_all_classes_but_one_path(dataset, class_removed):
	return read_pickle_path(dataset + '_features' + '_no_' + str(class_removed))

def get_data_obj_and_classifier(dataset_name, use_data_subset):
	return {
		MNIST : (MnistData(use_data_subset=use_data_subset), MnistClassifier()),
	
	}[dataset_name]

def generate_features(d, cls, dataset_name, class_removed=None):
	if class_removed is None:
		print("################# START -- generate features using all classes ##################")
	else:
		print("################# START -- generate features using all classes except %d ##################" % class_removed)

	d.set_removed_class(class_index=class_removed)
	cls.fit(*d.into_fit())

	predictor = cls.model
	extractor = Model(inputs=predictor.input,
	                 outputs=predictor.get_layer('features').output)

	d.set_removed_class(class_index=None)
	x, y = d._features_and_labels()
	features = extractor.predict(x)
	payload = {'features' : features, 'labels' : y}

	print("%d Feature vectors, %d labels" % (len(features), len(y)))

	if class_removed is None:
		write_path = write_features_from_all_classes_path(dataset_name)
	else:
		write_path = write_features_from_all_classes_but_one_path(dataset_name, class_removed)
	
	joblib.dump(payload, write_path)
	if class_removed is None:
		print("DONE -- generated features using all classes\n\n")
	else:
		print("DONE -- generated features using all classes except %d\n\n" % class_removed)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', help='what dataset_name to use, options: mnist xray cifar10', default='mnist')  # 
    parser.add_argument('-t', '--test', help='use small subset of data', action='store_true')
    args = parser.parse_args()

    main(dataset_name=args.dataset_name, use_data_subset=args.test)
