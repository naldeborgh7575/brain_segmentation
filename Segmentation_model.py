import numpy as np
import theano
from keras.models import Seqential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(5)

n_train_obs = X_train.shape[0]
n_test_obs = X_test.shape[0]
batch_size = 100
nb_classes = 5
nb_epoch = 10
img_width = 65
img_height = 65

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Seqential()
model.add(Convolution2D(n_filters, filter_w, filter_h, input_shape=(1, 4, 65, 65), activation='relu'))
