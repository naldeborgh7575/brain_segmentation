import numpy as np
import random
# import theano
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape, MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation, Flatten
from keras.regularizers import l1l2
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import classification_report

class BasicModel(object):
    def __init__(self, n_epoch=10, batch_size=32, single_or_dual='single'):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.single_or_dual = single_or_dual
        if self.single_or_dual == 'single':
            self.model_comp = self.compile_model()
        else:
            self.model_comp = self.comp_double()

    def compile_model(self):
        print 'Compiling model...'
        single = Sequential()
        single.add(Convolution2D(64, 7, 7, border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01), input_shape=(4,33,33)))
        single.add(Activation('relu'))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=256, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(Dropout(0.25))

        single.add(Flatten())
        single.add(Dense(5))
        single.add(Activation('softmax'))

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        single.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return single

    def comp_double(self):
        print 'Compiling model...'
        single = Sequential()
        single.add(Convolution2D(64, 7, 7, border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01), input_shape=(4,33,33)))
        single.add(Activation('relu'))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=256, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        single.add(Dropout(0.25))
        single.add(Flatten())

        # add small patch to train on
        five = Sequential()
        five.add(Reshape((100,1), input_shape = (4,5,5)))
        five.add(Flatten())
        five.add(MaxoutDense(128, nb_feature=5))
        five.add(Dropout(0.5))

        model = Sequential()
        # merge both paths
        model.add(Merge([five, single], mode='concat', concat_axis=1))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.001, decay=0.1, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return model

    def load_model_weights(self, model_name):
        pass

    def fit_model(self, X_train, y_train, X5_train = None):
        Y_train = np_utils.to_categorical(y_train, 5)

        shuffle = zip(X_train, Y_train)
        np.random.shuffle(shuffle)

        X_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

        if X5_train:
            self.model_comp.fit([X5_train, X_train], Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1)
        else:
            self.model_comp.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1)

    def class_report(self, X_test, y_test):
        '''
        INPUT   (1) list 'X_test': test data of 4x33x33 patches
                (2) list 'y_test': labels for X_test
        OUTPUT  (1) confusion matrix of precision, recall and f1 score
        '''
        y_pred = self.model_comp.predict_class(X_test)
        print classification_report(y_pred, y_test)

    def predict_image()



#older one, don't change

        # single.add(Convolution2D(nb_filter=256, nb_row=4, nb_col=4, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01)))
        # single.add(BatchNormalization(mode=0, axis=1))
        # single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        # single.add(Dropout(0.5))

# class BasicModel(object):
#     def __init__(self, single_or_cascade='single', n_epoch=10, batch_size=128):
#         self.single_or_cascade = single_or_cascade
#         self.n_epoch = n_epoch
#         self.batch_size = batch_size
#         self.model_comp = self.compile_model()
#
#     def compile_model(self):
#         print 'Compiling model...'
#         single = Sequential()
#         single.add(Convolution2D(64, 10, 10, border_mode='valid', input_shape=(4,33,33))) #, W_regularizer=l1l2(l1=0.01, l2=0.01)
#         single.add(Activation('relu'))
#         #single.add(Dropout(0.5))
#         single.add(Convolution2D(nb_filter=128, nb_row=7, nb_col=7, activation='relu', border_mode='valid')) #, W_regularizer=l1l2(l1=0.01, l2=0.01)
#         single.add(MaxPooling2D(pool_size=(4,4), strides=(1,1)))
#         #single.add(Dropout(0.5))
#         single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid')) #, W_regularizer=l1l2(l1=0.01, l2=0.01)
#         single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
#         single.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='valid')) #, W_regularizer=l1l2(l1=0.01, l2=0.01)
#         #single.add(Dropout(0.5))
#
#         single.add(Flatten())
#         single.add(Dense(5))
#         single.add(Activation('softmax'))
#
#         sgd = SGD(lr=0.0001, decay=0.1)
#         single.compile(loss='categorical_crossentropy', optimizer='adadelta')
#         print 'Done.'
#         return single
#
#     def fit_model(self, X_train, y_train):
#         Y_train = np_utils.to_categorical(y_train, 5)
#
#         shuffle = zip(X_train, Y_train)
#         np.random.shuffle(shuffle)
#
#         X_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
#         Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
#
#         self.model_comp.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1)

# accuracy reaching ~ 40 with updated architecture (local)
