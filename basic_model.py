import numpy as np
import random
# import theano
from keras.models import Graph, Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.core import Dropout, Activation, Flatten
from keras.regularizers import l1l2
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils

class BasicModel(object):
    def __init__(self, single_or_cascade='single', n_epoch=10, batch_size=128):
        self.single_or_cascade = single_or_cascade
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.model_comp = self.compile_model()

    def compile_model(self):
        print 'Compiling model...'
        single = Sequential()
        single.add(Convolution2D(64, 10, 10, border_mode='valid', input_shape=(4,33,33), W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(Activation('relu'))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=7, nb_col=7, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(MaxPooling2D(pool_size=(4,4), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(Dropout(0.5))

        single.add(Flatten())
        single.add(Dense(5))
        single.add(Activation('softmax'))

        sgd = SGD(lr=0.000001, decay=0.1)
        single.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return single

    def fit_model(self, X_train, y_train):
        Y_train = np_utils.to_categorical(y_train, 5)
        self.model_comp.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1)

 # val_acc ~ 40% (50000 samples, 10 epochs, batch size = 128)
 # lr of 0.01 = lower accuracy in 10 epochs with 2500 training samples
 # 1e-6 bounces around a lot, ends up around 30% (2500 samples)
 # try constraining the weights next
