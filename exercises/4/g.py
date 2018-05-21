'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import pandas as pd
import utils as ut

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 16
epochs = 50
epsilon_std = 1.0


def sampling(args):
    _z_mean, _z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(_z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return _z_mean + K.exp(_z_log_var / 2) * epsilon


input_shape = (28, 28, 1)
inputs = Input(shape=input_shape, name='encoder_input')

encoding_layers = [Conv2D(16, (3, 3), activation='relu', padding='same'),
                   MaxPooling2D((2, 2), padding='same'),
                   Conv2D(8, (3, 3), activation='relu', padding='same'),
                   MaxPooling2D((2, 2), padding='same')]

x = reduce(lambda x, a: a(x), encoding_layers, inputs)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

decoding_layers = [Dense(intermediate_dim, activation='relu'),
                   Dense(shape[1] * shape[2] * shape[3], activation='relu'),
                   Reshape((shape[1], shape[2], shape[3])),
                   Conv2D(8, (3, 3), activation='relu', padding='same'),
                   UpSampling2D((2, 2)),
                   Conv2D(16, (3, 3), activation='relu', padding='same'),
                   UpSampling2D((2, 2)),
                   Conv2D(1, (3, 3), activation='sigmoid', padding='same')]

outputs = reduce(lambda x, a: a(x), decoding_layers, z)

# instantiate VAE model
vae = Model(inputs, outputs, name='vae')

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((-1,) + input_shape)
x_test = x_test.reshape((-1,) + input_shape)

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

vae.save('g.vae.h5')

encoder = Model(inputs, z_mean)

x_encoded_mean = Input(shape=(latent_dim,))
x_decoded_mean = reduce(lambda x, a: a(x), decoding_layers, x_encoded_mean)
generator = Model(x_encoded_mean, x_decoded_mean)

# save the models
encoder.save('g.encoder.h5')
generator.save('g.generator.h5')

# (c)
print('\nCorresponding mapping coordinates in the latent space - one image per digit:\n')
encoded_x_test = encoder.predict(x_test)
digits = {}
i = 0
while len(digits) < 10:
    digits[y_test[i]] = encoded_x_test[i]
    i += 1

ut.output('g.c', pd.DataFrame(digits))

# (d)
z_sample = np.array([[-2.5, 0.55]])
decoded_x = generator.predict(z_sample)
ut.image(decoded_x)
ut.plt_save('g.d')

# (e)
source, target = digits[3], digits[5]
(x1, y1), (x2, y2) = source, target
a = (y2 - y1) / (x2 - x1)
b = y1 - (a * x1)
f = lambda _x_: a * _x_ + b
x_samples = np.linspace(x1, x2, num=10)
samples = [np.array([[_x, f(_x)]]) for _x in x_samples]
ut.images(map(generator.predict, samples), map(str, x_samples), n_cols=3)
ut.plt_save('g.e')
