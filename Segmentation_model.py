import numpy as np
import random
# import theano
from keras.models import Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(5)

class KerasModel():
    def __init__(self, h_1=65, w_1=65, h_2=33, w_2=33, n_filters = [64, 64, 160, 5], n_classes = 5, n_chan = 4, n_epoch = 3, batch_size = 32, pool_size_1 = 4, pool_size_2 = 2, local_conv_1 = 7, local_conv_2 = 3, global_conv = 13, output_conv = 21, learning_rate = 0.005, decay_factor = 0.1, momentum_coef_1 = 0.5, momentum_coef_2 = 0.9):
        self.h_1 = h_1
        self.h_2 = h_2
        self.w_1 = w_1
        self.w_2 = w_2
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
        self.momentum_coef_2 = momentum_coef_2
        self.model_comp = self.compile_model()
        # self.n_train_obs = X_train.shape[0]
        # self.n_test_obs = X_test.shape[0]

    def compile_model(self):
        model = Graph()

        # input layer: 65 x 65 patch. split to local and global paths
        model.add_input(name='input_1', input_shape=(self.n_chan, self.h_1, self.w_1), batch_input_shape=(self.batch_size, self.n_chan, self.h_1, self.w_1), dtype='float')

        # First local conv layer: 7x7 filter + 4x4 maxpooling
        model.add_node(Dropout(0.9), name='input_dropout', input='input_1')
        model.add_node(Convolution2D(nb_filter=self.n_filters[0], nb_row=self.local_conv_1, nb_col=self.local_conv_1, border_mode='valid', activation='relu', input_shape=(4,65,65)), name='local_k11', input='input_dropout')
        model.add_node(MaxPooling2D(pool_size=(self.pool_size_1, self.pool_size_1), strides = (1,1), border_mode='valid'), name='local_p11', input='local_k11')

        # Second local conv layer: 3x3 filter + 2x2 maxpooling
        model.add_node(Dropout(0.75), name='drop_p11', input='local_p11')
        model.add_node(Convolution2D(nb_filter = self.n_filters[1], nb_row=self.local_conv_2, nb_col=self.local_conv_2, activation='relu', border_mode='valid'), name='local_k21', input='drop_p11')
        model.add_node(MaxPooling2D(pool_size = (self.pool_size_2, self.pool_size_2), strides=(1,1), border_mode='valid'), name='local_p21', input='local_k21')

        # First global conv layer: 13x13 filter, no maxpooling
        model.add_node(Dropout(0.75), name='drop_g1', input='input_1')
        model.add_node(Convolution2D(nb_filter=self.n_filters[2], nb_row=self.global_conv, nb_col=self.global_conv, border_mode='valid', activation='relu'), name='global_1', input='drop_g1')
        # print self.n_filters[2], self.global_conv

        # Concat local and global nodes
        model.add_node(Dropout(0.5), name='drop_lp21', input='local_p21')
        model.add_node(Dropout(0.5), name='drop_global', input='global_1')
        model.add_node(Convolution2D(nb_filter=self.n_filters[3], nb_row=self.output_conv, nb_col=self.output_conv, border_mode='valid', activation='relu'), name='merge_1', inputs=['drop_lp21', 'drop_global'], merge_mode='concat', concat_axis=1)

        # Input for second net in cascade
        model.add_input(name='input_2', input_shape=(self.n_chan, self.h_2, self.w_2))

        # Second local conv layer: merge 33x33 input with first net output
        model.add_node(Dropout(0.9), name='input2_drop', input='input_2')
        model.add_node(Dropout(0.75), name='merge_drop', input='merge_1')
        model.add_node(Convolution2D(nb_filter=self.n_filters[0], nb_row=self.local_conv_1, nb_col=self.local_conv_1, border_mode='valid', activation='relu'), name='local_k12', inputs=['merge_drop', 'input2_drop'], merge_mode='concat', concat_axis=1)
        model.add_node(MaxPooling2D(pool_size=(self.pool_size_1, self.pool_size_1), strides=(1,1), border_mode='valid'), name='local_p12', input='local_k12')

        # Second conv of second local conv layer
        model.add_node(Dropout(0.75), name='drop_lp12', input='local_p12')
        model.add_node(Convolution2D(nb_filter=self.n_filters[1], nb_row=self.local_conv_2, nb_col=self.local_conv_2, border_mode='valid', activation='relu'), name='local_k22', input='drop_lp12')
        model.add_node(MaxPooling2D(pool_size=(self.pool_size_2, self.pool_size_2), strides=(1,1), border_mode='valid'), name='local_p22', input='local_k22')

        # Merge 33x33 input with forst net output, send to global path
        model.add_node(Convolution2D(nb_filter=self.n_filters[2], nb_row=self.global_conv, nb_col=self.global_conv, border_mode='valid', activation='relu'), name='global_2', inputs=['merge_drop', 'input2_drop'], merge_mode='concat', concat_axis=1)

        # Concat local and global filters, output = 5x1x1
        model.add_node(Dropout(0.5), name='drop_lp22', input='local_p22')
        model.add_node(Dropout(0.5), name='drop_global22', input='global_2')
        model.add_node(Convolution2D(nb_filter=self.n_filters[3], nb_row=self.output_conv, nb_col=self.output_conv, border_mode='valid', activation='relu'), name='merge_2', inputs=['drop_lp22', 'drop_global22'], merge_mode='concat', concat_axis=1)

        model.add_node(Flatten(), name='flatten', input='merge_2')
        model.add_node(Dense(5, init='uniform', activation='softmax'), name='dense_output', input='flatten')
        model.add_output(name='output_final', input='dense_output')

        model.compile('sgd', loss={'output_final':'mse'})
        return model

    def fit_model(self, X_train, X_33_train, y_train):
        Y_train = np_utils.to_categorical(y_train, 5)
        X_train = X_train.astype("int")
        X_33_train = X_33_train.astype('int')
        prep = zip(X_train, X_33_train, Y_train)
        random.shuffle(prep)

        X_train = np.array([prep[i][0] for i in xrange(len(prep))])
        X33_train = np.array([prep[i][1] for i in xrange(len(prep))])
        Y_train = np.array([prep[i][2] for i in xrange(len(prep))])

        data = {'input_1': X_train, 'input_2': X33_train, 'output_final': Y_train}

        self.model_comp.fit(data, batch_size=32, nb_epoch = self.n_epoch, show_accuracy = True, verbose=1, validation_split=0.1)
        # score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
