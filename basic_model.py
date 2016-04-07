import numpy as np
import random
# import theano
from keras.models import Graph, Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Dropout, Activation, Flatten
from keras.regularizers import l1l2
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils

model = Sequential()




model.add(Convolution2D(nb_filter = 64, nb_row=15, nb_col=15, input_shape=(4,65,65), activation='relu', border_mode='valid'))
model.add(Convolution2D(nb_filter=64, nb_row=10, nb_col=10, activation='relu', border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('softmax'))

sgd = SGD(lr=0.005, decay=0.1, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer='sgd')
