'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import pandas as pd
import utils as ut

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    _z_mean, _z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(_z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return _z_mean + K.exp(_z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# new code starts here
encoder = Model(x, z_mean)

x_encoded_mean = Input(shape=(latent_dim,))
_h_decoded = decoder_h(x_encoded_mean)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(x_encoded_mean, _x_decoded_mean)

# save the models
# vae.save('vae.h5')
# encoder.save('encoder.h5')
# generator.save('generator.h5')

# (c)
colors = ['red', 'blue', 'green', 'lightsalmon', 'pink', 'purple', 'orange', 'black', 'brown', 'magenta']
latent = encoder.predict(x_test)
for digit in xrange(10):
    X = latent[y_test == digit]
    x1, x2 = [x[0] for x in X], [x[1] for x in X]
    plt.scatter(x1, x2, alpha=1, s=1, c=colors[digit], label='{0}\'s'.format(digit))

ut.plot(name='c',
        title='Test Set Visualization',
        xlabel='Latent dim 1',
        ylabel='Latent dim 2')

print('\nCorresponding mapping coordinates in the latent space - one image per digit:\n')
encoded_x_test = encoder.predict(x_test)
digits = {}
i = 0
while len(digits) < 10:
    digits[y_test[i]] = encoded_x_test[i]
    i += 1

ut.output('c', pd.DataFrame(digits))

# (d)
z_sample = np.array([[-2.5, 0.55]])
decoded_x = generator.predict(z_sample)
ut.image(decoded_x)
ut.plt_save('d')

# (e)
source, target = digits[3], digits[5]
(x1, y1), (x2, y2) = source, target
a = (y2 - y1) / (x2 - x1)
b = y1 - (a * x1)
f = lambda _x_: a * _x_ + b
x_samples = np.linspace(x1, x2, num=10)
samples = [np.array([[_x, f(_x)]]) for _x in x_samples]
ut.images(map(generator.predict, samples), map(str, x_samples), n_cols=3)
ut.plt_save('e')
