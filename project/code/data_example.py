from classifier import Classifier
from generator import LowShotGenerator
from data_utils import *

import pandas as pd

# do this once! this method takes time!
data_obj = get_processed_data(num_files_to_fetch_data_from=1)

# 1. fetching the classifier from file:
cls = Classifier(model_weights_file_path=model_path('model'))

# 2. evaluating the classifier on some data:
print("\n2. evaluating the classifier on some data:")
X, y = get_features_and_labels(data_obj)
X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.99)
scores = cls.evaluate(X_test, y_test)
print("accuracy is %f" % scores[1])

# 3. train a model on some subset of diseases and save model weights to new file:
print("\n3. train a model on some subset of diseases and save model weights to new file:")
diseases_to_remove = ['Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
X, y = remove_diseases(X, y, diseases_to_remove, data_obj)
X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.1)

cls = Classifier(n_classes=N_CLASSES - len(diseases_to_remove))
cls.fit(X_train, y_train, model_weights_file_path=model_path('some_new_model'))
loss, acc = cls.evaluate(X_test, y_test)
print("accuracy acheived: %f" % acc)

# all diseases classes:
print("\nAll possible disease classes:")
print(pd.DataFrame(data_obj['label_encoder_classes']))
