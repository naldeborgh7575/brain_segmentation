import numpy as np
# import theano
from keras.models import Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(5)

class KerasModel():
    def __init__(self, h=65, w=65, n_filters = [64, 64, 160, 5], n_classes = 5, n_chan = 4, n_epoch = 10, batch_size = 32, pool_size_1 = 4, pool_size_2 = 2, local_conv_1 = 7, local_conv_2 = 3, global_conv = 13, output_conv = 21, learning_rate = 0.005, decay_factor = 0.1, momentum_coef_1 = 0.5, momentum_coef_2 = 0.9):
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
        self.momentum_coef_2 = momentum_coef_2
        # self.n_train_obs = X_train.shape[0]
        # self.n_test_obs = X_test.shape[0]

    def compile_model(self):
        model = Graph()

        # input layer: 65 x 65 patch. split to local and global paths
        model.add_input(name='input_1', input_shape=(self.n_chan, self.h, self.w), batch_input_shape=(self.batch_size, self.n_chan, self.h, self.w), dtype='float')

        # First local conv layer: 7x7 filter + 4x4 maxpooling
        model.add_node(Convolution2D(nb_filter=self.n_filters[0], nb_row=self.local_conv_1, nb_col=self.local_conv_1, border_mode='same', activation='relu', input_shape=(4,65,65)), name='local_k11', input='input_1')
        model.add_node(MaxPooling2D(pool_size=(self.pool_size_1, self.pool_size_1)), name='local_p11', input='local_k11')

        # Second local conv layer: 3x3 filter + 2x2 maxpooling
        model.add_node(Convolution2D(nb_filter = self.n_filters[1], nb_row=self.local_conv_2, nb_col=self.local_conv_2, activation='relu', input_shape=(64,56,56)), name='local_k21', input='local_p11')
        model.add_node(MaxPooling2D(pool_size = (self.pool_size_2, self.pool_size_2)), name='local_p21', input='local_k21')

        # First global conv layer: 13x13 filter, no maxpooling
        model.add_node(Convolution2D(nb_filter=self.n_filters[2], nb_row=self.global_conv, nb_col=self.global_conv, border_mode='same', activation='relu', input_shape=(4,65,65)), name='global_1', input='input_1')
        # print self.n_filters[2], self.global_conv

        # Concat local and global nodes
        model.add_shared_node(Convolution2D(nb_filter=self.n_filters[3], nb_row=self.output_conv, nb_col=self.output_conv, activation='relu', input_shape = (224,53,53)), name='merge_1', inputs=['local_p21', 'global_1'], merge_mode='concat', concat_axis=-1)

        # Create output node for first net in cascade
        model.add_output(name='output_1', input='merge_1')

        self.model_comp = model.compile(optimizer = 'SGD', loss={'output_1':'categorical_crossentropy'})

    def fit_model(self, X_train, y_train):
        Y_train = np_utils.to_categorical(y_train, 5)
        # Y_test = np_utils.to_categorical(y_test, 5)
        X_train = X_train.astype("float32")
        # X_test = X_test.astype("float32")
        if np.max(X_train != 0):
            X_train /= np.max(X_train)
        # if np.max(X_test) != 0:
            # X_test /= np.max(X_test)

        self.model_comp.fit(X_train, Y_train, batch_size = self.batch_size, nb_epoch = self.n_epoch, show_accuracy = True, verbose=1, validation_split=0.1)
        # score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])
