from data_object import *
from mnist_data import *
from mnist_classifier import *
from cifar_data import *
from cifar_classifier import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.externals import joblib
from data_utils import *
from cifar_data import *
import argparse
from keras.models import model_from_json

MNIST = 'mnist'
CIFAR10 = 'cifar10'
XRAY = 'xray'

def main(dataset_name, use_data_subset=False, epochs=1):
	
	d, cls = get_data_obj_and_classifier(dataset_name=dataset_name, 
										 use_data_subset=use_data_subset,
										 epochs=epochs)
	# generate using all classes
	generate_features(d, cls, dataset_name, class_removed=None)

	# generate ommiting each class in turn
	n_classes = d.get_num_classes()
	for class_index in range(n_classes):
		generate_features(d, cls, dataset_name, class_removed=class_index)		

def get_data_obj_and_classifier(dataset_name, use_data_subset, epochs):
	return {
		MNIST : (MnistData(use_data_subset=use_data_subset), MnistClassifier(epochs=epochs)),
		CIFAR10 : (Cifar10Data(use_data_subset=use_data_subset), Cifar10Classifier(epochs=epochs))
	}[dataset_name]

def load_model(dataset, class_removed = None):
	model_name = generate_model_name(dataset, class_removed = class_removed)
	json_file = open(read_model_json_path(model_name), 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(read_model_path(model_name))
	
	print("Loaded model from disk")
	return loaded_model

def generate_model_name(dataset_name, class_removed = None):
	return dataset_name + '_' + str(class_removed)

def generate_features(d, cls, dataset_name, class_removed=None):
	if class_removed is None:
		print("################# START -- generate features using all classes ##################")
	else:
		print("################# START -- generate features using all classes except %d ##################" % class_removed)

	d.set_removed_class(class_index=class_removed)
	cls.fit(*d.into_fit())

	model_name = generate_model_name(dataset_name, class_removed = class_removed)

	with open(write_model_json_path(model_name), "w") as json_file:
	    json_file.write(cls.model.to_json())

	cls.model.save_weights(write_model_path(model_name))
	print("Saved model to disk")

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
    parser.add_argument('-e', '--epochs', help='number if epochs', type=int, default=1)
    args = parser.parse_args()

    main(dataset_name=args.dataset_name, use_data_subset=args.test, epochs=args.epochs)
