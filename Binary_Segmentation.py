import numpy as np
import random
# import theano
from keras.models import Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Dropout, Activation, Flatten
from keras.regularizers import l1l2
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
np.random.seed(5)

class KerasBinary(object):
    def __init__(self, h=33, w=33, n_filters = [64, 64, 160, 2], n_classes = 2, n_chan = 4, n_epoch = 3, batch_size = 12, pool_size_1 = 4, pool_size_2 = 2, local_conv_1 = 7, local_conv_2 = 3, global_conv = 13, output_conv = 21, learning_rate = 0.005, decay_factor = 0.1, momentum_coef_1 = 0.5):
        self.h = h
        self.w = w
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.n_chan = n_chan
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.local_conv_1 = local_conv_1
        self.local_conv_2 = local_conv_2
        self.global_conv = global_conv
        self.output_conv = output_conv
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.momentum_coef_1 = momentum_coef_1
        # self.momentum_coef_2 = momentum_coef_2
        self.model_comp = self.compile_model()
        # self.n_train_obs = X_train.shape[0]
        # self.n_test_obs = X_test.shape[0]

    def compile_model(self):
        model = Graph()

        # input layer: 33 x 33 patch. split to local and global paths
        model.add_input(name='input_1', input_shape=(self.n_chan, self.h, self.w), batch_input_shape=(self.batch_size, self.n_chan, self.h, self.w), dtype='float')
        # First local conv layer: 7x7 filter + 4x4 maxpooling
        model.add_node(Dropout(0.75), name='input_dropout', input='input_1')
        model.add_node(Convolution2D(nb_filter=self.n_filters[0], nb_row=self.local_conv_1, nb_col=self.local_conv_1, border_mode='valid', activation='relu', W_regularizer=l1l2(l1=0.01, l2=0.01), W_constraint=maxnorm(2)), name='local_k1', input='input_dropout')
        model.add_node(MaxPooling2D(pool_size=(self.pool_size_1, self.pool_size_1), strides = (1,1), border_mode='valid'), name='local_p1', input='local_k1')
        # Second local conv layer: 3x3 filter + 2x2 maxpooling
        model.add_node(Dropout(0.75), name='drop_p1', input='local_p1')
        model.add_node(Convolution2D(nb_filter = self.n_filters[1], nb_row=self.local_conv_2, nb_col=self.local_conv_2, activation='relu', border_mode='valid', W_regularizer=l1l2(l1=0.01, l2=0.01), W_constraint=maxnorm(2)), name='local_k2', input='drop_p1')
        model.add_node(MaxPooling2D(pool_size = (self.pool_size_2, self.pool_size_2), strides=(1,1), border_mode='valid'), name='local_p2', input='local_k2')
        # First global conv layer: 13x13 filter, no maxpooling
        model.add_node(Dropout(0.75), name='drop_g', input='input_1')
        model.add_node(Convolution2D(nb_filter=self.n_filters[2], nb_row=self.global_conv, nb_col=self.global_conv, border_mode='valid', activation='relu', W_regularizer=l1l2(l1=0.01, l2=0.01), W_constraint=maxnorm(2)), name='global', input='drop_g')
        # Concat local and global nodes
        model.add_node(Dropout(0.5), name='drop_lp2', input='local_p2')
        model.add_node(Dropout(0.5), name='drop_global', input='global')
        model.add_node(Convolution2D(nb_filter=self.n_filters[3], nb_row=self.output_conv, nb_col=self.output_conv, border_mode='valid', activation='relu', W_regularizer=l1l2(l1=0.01, l2=0.01), W_constraint=maxnorm(2)), name='merge', inputs=['drop_lp2', 'drop_global'], merge_mode='concat', concat_axis=1)
        # Flatten output of 5x1x1 to 1x5, perform softmax
        model.add_node(Flatten(), name='flatten', input='merge')
        model.add_node(Dense(2, init='uniform', activation='softmax'), name='dense_output', input='flatten')
        model.add_output(name='output_final', input='dense_output')

        sgd = SGD(lr=0.01, decay=0.1, nestrov=True)
        model.compile('sgd', loss={'output_final':'categorical_crossentropy'})
        return model

    def fit_model(self, X_train, y_train):
        Y_train = np_utils.to_categorical(y_train, 2)
        X_train = X_train.astype("int")
        prep = zip(X_train, Y_train)
        random.shuffle(prep) #randomize inputs

        X_train = np.array([prep[i][0] for i in xrange(len(prep))])
        Y_train = np.array([prep[i][1] for i in xrange(len(prep))])

        data = {'input_1': X_train, 'output_final': Y_train}

        self.model_fit = self.model_comp.fit(data, batch_size=self.batch_size, nb_epoch = self.n_epoch, show_accuracy = True, verbose=1, validation_split=0.1)
