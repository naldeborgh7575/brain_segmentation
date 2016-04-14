import numpy as np
import random
import matplotlib.pyplot as plt
import skimage.io as io
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report
from keras.models import Sequential, Graph, model_from_json
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape, MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

class BasicModel(object):
    def __init__(self, n_epoch=10, n_chan=4, batch_size=128, loaded_model=False, single_or_dual='single', w_reg=0.01):
        '''
        INPUT
        '''
        self.n_epoch = n_epoch
        self.n_chan = n_chan
        self.batch_size = batch_size
        self.single_or_dual = single_or_dual
        self.loaded_model = loaded_model
        self.w_reg = w_reg
        if not self.loaded_model:
            if self.single_or_dual == 'two_path':
                self.model_comp = self.comp_two_path()
            elif self.single_or_dual == 'dual':
                self.model_comp = self.comp_double()
            else:
                self.model_comp = self.compile_model()

    def compile_model(self):
        print 'Compiling single model...'
        single = Sequential()

        single.add(Convolution2D(64, 7, 7, border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg), input_shape=(self.n_chan,33,33)))
        single.add(Activation('relu'))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=5, nb_col=5, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=self.w_reg, l2=self.w_reg)))
        single.add(Dropout(0.25))

        single.add(Flatten())
        single.add(Dense(5))
        single.add(Activation('softmax'))

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        single.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return single

    def comp_two_path(self):
        print 'Compiling two-path model...'
        model = Graph()
        model.add_input(name='input', input_shape=(self.n_chan, 33, 33))

        # local pathway, first convolution/pooling
        model.add_node(Convolution2D(64, 7, 7, border_mode='valid', activation='relu', W_regularizer=l1l2(l1=0.01, l2=0.01)), name='local_c1', input= 'input')
        model.add_node(MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid'), name='local_p1', input='local_c1')

        # local pathway, second convolution/pooling
        model.add_node(Dropout(0.5), name='drop_lp1', input='local_p1')
        model.add_node(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', W_regularizer=l1l2(l1=0.01, l2=0.01)), name='local_c2', input='drop_lp1')
        model.add_node(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid'), name='local_p2', input='local_c2')

        # global pathway
        model.add_node(Convolution2D(160, 13, 13, border_mode='valid', activation='relu', W_regularizer=l1l2(l1=0.01, l2=0.01)), name='global', input='input')

        # merge local and global pathways
        model.add_node(Dropout(0.5), name='drop_lp2', input='local_p2')
        model.add_node(Dropout(0.5), name='drop_g', input='global')
        model.add_node(Convolution2D(5, 21, 21, border_mode='valid', activation='relu',  W_regularizer=l1l2(l1=0.01, l2=0.01)), name='merge', inputs=['drop_lp2', 'drop_g'], merge_mode='concat', concat_axis=1)

        # Flatten output of 5x1x1 to 1x5, perform softmax
        model.add_node(Flatten(), name='flatten', input='merge')
        model.add_node(Dense(5, activation='softmax'), name='dense_output', input='flatten')
        model.add_output(name='output', input='dense_output')

        sgd = SGD(lr=0.005, decay=0.1, momentum=0.9)
        model.compile('sgd', loss={'output':'categorical_crossentropy'})
        print 'Done.'
        return model

    def comp_double(self):
        print 'Compiling double model...'
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

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        print 'Done.'
        return model

    def load_model_weights(self, model_name):
        '''
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        model = '{}.json'.format(model_name)
        weights = '{}.h5'.format(model_name)
        with open(model_n) as f:
            m = f.next()
        self.model_load = model_from_json(json.loads(m))

    def fit_model(self, X_train, y_train, X5_train = None):
        '''
        INPUT   (1) numpy array 'X_train': list of patches to train on in form (n_sample, n_channel, h, w)
                (2) numpy vector 'y_train': list of labels corresponding to X_train patches in form (n_sample,)
                (3) numpy array 'X5_train': center 5x5 patch in corresponding X_train patch. if None, uses single-path architecture
        OUTPUT  (1) Fits specified model
        '''
        Y_train = np_utils.to_categorical(y_train, 5)

        shuffle = zip(X_train, Y_train)
        np.random.shuffle(shuffle)

        X_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

        # Save model after each epoch to check/bm_epoch#-val_loss
        checkpointer = ModelCheckpoint(filepath="./check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

        if self.single_or_dual == 'dual':
            self.model_comp.fit([X5_train, X_train], Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        elif self.single_or_dual == 'two_path':
            data = {'input': X_train, 'output': Y_train}
            self.model_comp.fit(data, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1, callbacks=[checkpointer])
        else:
            self.model_comp.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_split=0.1, show_accuracy=True, verbose=1, callbacks=[checkpointer])

    def save_model(self, model_name):
        pass

    def class_report(self, X_test, y_test):
        '''
        INPUT   (1) list 'X_test': test data of 4x33x33 patches
                (2) list 'y_test': labels for X_test
        OUTPUT  (1) confusion matrix of precision, recall and f1 score
        '''
        y_pred = self.model_comp.predict_class(X_test)
        print classification_report(y_pred, y_test)

    def predict_image(self, test_img, show=True):
        imgs = io.imread(test_img).astype('float').reshape(5,240,240)
        plist = []

        # create patches from an entire slice
        for img in imgs[:-1]:
            if np.max(img) != 0:
                img /= np.max(img)
            p = extract_patches_2d(img, (33,33))
            plist.append(p)
        patches = np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))

        # predict classes of each pixel based on model
        full_pred = self.model_comp.predict_classes(patches)
        fp1 = full_pred.reshape(208,208)
        if show:
            io.imshow(fp1)
            plt.show


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
