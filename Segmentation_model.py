import numpy as np
import theano
from keras.models import Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(5)

class KerasModel():
    def __init__(self, h=65, w=65, n_filters = 5, n_classes = 5, n_chan = 4, n_epoch = 10, batch_size = 32, pool_size_1 = 4, pool_size_2 = 2, local_conv_1 = 7, local_conv_2 = 3, global_conv = 13, output_conv = 21, learning_rate = 0.005, decay_factor = 0.1, momentum_coef_1 = 0.5, momentum_coef_2 = 0.9):
        self.h = h
        self.w = w
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.n_chan = n_chan
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
        self.n_train_obs = X_train.shape[0]
        self.n_test_obs = X_test.shape[0]

    def compile_model(self):
        model = Graph()

        # input layer: 65 x 65 patch. split to local and global paths
        model.add_input(name='input_1', input_shape=(self.n_chan, self.h, self.w), batch_input_shape=(self.batch_size, self.n_chan, self.h, self.w), dtype='float')

        # First local conv layer: 7x7 filter + 4x4 maxpooling
        model.add_node(Convolution2D(self.n_filters, self.local_conv_1, self.local_conv_1, border_mode='same', input_shape=(self.n_chan, self.h, self.w), activation='relu'), name='local_k11', input='input_1')
        model.add_node(MaxPooling2D(pool_size=(self.pool_size_1, self.pool_size_1)), name='local_p11', input='local_k11')

        # Second local conv layer: 3x3 filter + 2x2 maxpooling
        model.add_node(Convolution2D(self.n_filters, self.local_conv_2, self.local_conv_2, activation='relu'), name='local_k21', input='local_p11')
        model.add_node(MaxPooling2D(pool_size = (self.pool_size_2, self.pool_size_2)), name='local_p21', input='local_k21')

        # First global conv layer: 13x13 filter, no maxpooling
        model.add_node(Convolution2D(self.n_filters, self.global_conv, self.global_conv, border_mode='same', input_shape=(self.n_chan, self.h, self.w), activation='relu'), name='global_1', input='input_1')

        # Concat local and global nodes
        model.add_node(Convolution2D(self.n_filters, self.output_conv, self.output_conv, activation='softmax'), name='merge_1' inputs=['global_1', 'local_p21'], merge_mode='concat', concat_axis=-1)

        # Create output node for first net in cascade
        model.add_output(name='output_1', input='merge_1')
