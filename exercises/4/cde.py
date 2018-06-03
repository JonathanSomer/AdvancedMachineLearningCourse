import numpy as np
import pandas as pd
import utils as ut
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.datasets import mnist

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# for sections c, d, e + f
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# for section g
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((-1, 28, 28, 1))
# x_test = x_test.reshape((-1, 28, 28, 1))

encoder = load_model('models/f.encoder.h5')
generator = load_model('models/f.generator.h5')

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
