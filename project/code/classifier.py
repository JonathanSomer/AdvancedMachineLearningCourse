from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling3D, AveragePooling2D
from keras.optimizers import Adam


class Classifier(object):
	def __init__(self, n_classes=15, model_weights_file_path=None):
		
		self.model = self.new_model(n_classes)

		if model_weights_file_path is not None:
			self.model.load_weights(model_weights_file_path)
		
		optimizer = self.get_optimizer()		
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	def new_model(self, n_classes):
	    a = Input(shape=(7, 7, 2048,))
	    b = GlobalMaxPooling2D()(a)
	    b = Dense(n_classes, activation='softmax')(b)
	    return Model(a,b)

	def get_optimizer(self):
		return Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	# supply file_path if want to save model to file
	def fit(self, X_train, y_train, model_weights_file_path=None):
	    self.model.fit(X_train, y_train, batch_size=50, epochs=10, verbose=2, validation_split=0.1)

	    if model_weights_file_path is not None:
	        self.model.save(model_weights_file_path)
	        
	    return self.model

	 # returns an array with [loss, accuracy]
	def evaluate(self, X_test, y_test):
		return self.model.evaluate(X_test, y_test, batch_size=32, verbose=1)
