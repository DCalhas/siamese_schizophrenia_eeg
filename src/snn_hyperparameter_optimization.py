from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import math

import random
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D, dot, BatchNormalization
from keras.backend import dot, sqrt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import RMSprop, Adam, SGD
from keras import regularizers
from keras.losses import hinge
from keras import backend as K

from scipy.signal import stft
from scipy.signal import get_window
from scipy.spatial.distance import cosine, chebyshev

import GPy
import GPyOpt

import eeg
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

num_classes = 2
epochs = 20
number_channels = 16


frequencies_nomenclature = ['delta', 'theta1', 'theta2', 'alpha1', 'alpha2', 
                   'beta1', 'beta2', 'beta3', 'gamma']

init = keras.initializers.glorot_uniform(seed=0)

def stft_to_freq_bin(freq_map, frequencies, timesteps):
    #at each time step binarize the frequencies freq_map[:,0]
    freq_map = abs(freq_map)
    
    stft_bins = []
    
    for timestep in range(len(timesteps)):
        bins = [0]*9
        for freq in range(len(frequencies)):
            if(frequencies[freq] <= 4):
                bins[0] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=6):
                bins[1] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=8):
                bins[2] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=10):
                bins[3] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=13):
                bins[4] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=16):
                bins[5] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=23):
                bins[6] += freq_map[:,timestep][freq]
            elif(frequencies[freq] <=30):
                bins[7] += freq_map[:,timestep][freq]
            elif(frequencies[freq] >30):
                bins[8] += freq_map[:,timestep][freq]
        
        stft_bins += [bins]
    
    return np.array(stft_bins)



def get_population_stft_by_bins(population, ids, binarize=True, n_channels=16, fs=128, nperseg=256):
    pop_stft = []
    
    for idt in ids:
        individual = population[idt]
        ind_channels_stft = []
        for channel in range(n_channels):
            f, t, Zxx = stft(individual[:,channel], fs, 
            nperseg=nperseg)#, window='hann')

            if(binarize):
                Zxx = stft_to_freq_bin(Zxx, f, t)
            else:
                Zxx = abs(Zxx)

            ind_channels_stft += [Zxx]
        pop_stft += ind_channels_stft

    return pop_stft

def cosine_distance(vects):
    x,y = vects
    #how should the normalization be done??
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    a = K.batch_dot(x, y, axes=1)

    b = K.batch_dot(x, x, axes=1)
    c = K.batch_dot(y, y, axes=1)

    return 1 - (a / (K.sqrt(b) * K.sqrt(c)))
    #line below is correct
    #return K.mean(1-K.abs(K.batch_dot(x, y, axes=1)))

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

class contrastive_loss():
    def __init__(self, margin):
        self.margin = margin

    def loss(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, y, n_channels=16):
    pairs = []
    labels = []
    #interval in order to channels pair with same channels
    interval = int(len(x)/(len(x)/n_channels))
    for ind in range(0, len(x), interval):
        #condition in order for the for not to exceed length
        if(int(len(x[ind:])/n_channels) > 1):
            #inner interval as the individual increases
            inner_interval = int(len(x[ind+n_channels:])/(len(x[ind+n_channels:])/n_channels))
            for other_ind in range(ind+n_channels, len(x), inner_interval):
                for channel in range(n_channels):
                    pairs += [[x[ind + channel], x[other_ind + channel]]]
                    if(y[ind + channel] == y[other_ind + channel]):
                        labels += [1]
                    else:
                        labels += [0]
            
    pairs = np.array(pairs)
    labels = np.array(labels)

    return pairs, labels


def create_base_network(input_shape, kernel_size=(6,6), final_dimension=12, regularization=0.011):
    ##model building
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    #flatten since too many dimensions, we only want a classification output
    model.add(Conv2D(1, kernel_size=kernel_size,
                     activation='relu',
                     input_shape=input_shape, kernel_initializer=init,
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l1(regularization),#0.011
                bias_regularizer=regularizers.l1(regularization)))#0.011
    
    model.add(Dropout(0.5))
    model.add(Conv2D(1, kernel_size=kernel_size,
                    activation='relu', kernel_initializer=init,
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l1(regularization),#0.011
                bias_regularizer=regularizers.l1(regularization)))#0.011
    
    model.add(Dropout(0.5))
    #things to test in order to increase the performance of the mdel
    #play a little with the kernel sizes - test values: (6,6)
    #change the optimization function - 
    model.add(Flatten())
    #embedding sizes with better results seem to be between [8,15[
    model.add(Dense(final_dimension, activation='softmax', kernel_initializer=init,
                bias_initializer='zeros'))#13
    print(model.summary())
    return model

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class siamese:
    def __init__(self, input_shape, regularization=0.011, kernel_size=(6,6), final_dimension=12, learning_rate=0.0004, margin=1.2):
        self.base_network = create_base_network(input_shape, kernel_size, final_dimension, regularization)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)

        distance = Lambda(cosine_distance,#compare this results with euclidean
                          output_shape=cos_dist_output_shape)([processed_a, 
                          processed_b])

        model = Model([input_a, input_b], distance)

        adam = Adam(lr=learning_rate)
        loss_function = contrastive_loss(margin)
        model.compile(loss=loss_function.loss, optimizer=adam, metrics=[accuracy])
        self.model = model

    def save_base_network(self, k):
        (self.base_network).save_weights("base_network_partition_" + str(k) + ".h5")

#early stopping based exclusively on the training loss value
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            self.model.stop_training = True


def data(population, y_train, y_test, normalization_factor=0.0):
    binarize=False
    nperseg = 512

    X_train = np.array(get_population_stft_by_bins(population, y_train, binarize=binarize, 
        n_channels=number_channels, nperseg=nperseg))

    y = []
    for i in range(len(y_train)):
        if(y_train[i] >= 39):
            y += [1]*number_channels
        else:
            y += [0]*number_channels
    y_train = y
     
    X_test = np.array(get_population_stft_by_bins(population, y_test, binarize=binarize, 
        n_channels=number_channels, nperseg=nperseg))

    y = []
    for i in range(len(y_test)):
        if(y_test[i] >= 39):
            y += [1]*number_channels
        else:
            y += [0]*number_channels
    y_test = y

    time = len(X_train[0])
    freq = len(X_train[0][0])
    X_train = X_train.reshape(X_train.shape[0], time, freq, 1)
    X_test = X_test.reshape(X_test.shape[0], time, freq, 1)
    input_shape = (time, freq, 1)
    #more reshaping
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #normalize
    if(normalization_factor > 0.0):
        normalize_features(X_train, max_magnitude=normalization_factor)
        normalize_features(X_test, max_magnitude=normalization_factor)

    # create training+test positive and negative pairs
    tr_pairs, tr_y = create_pairs(X_train, y_train, n_channels=number_channels)

    te_pairs, te_y = create_pairs(X_test, y_test, n_channels=number_channels)

    return tr_pairs, tr_y, te_pairs, te_y, input_shape

#normalize the frequency amplityudes until a certain threshold
def normalize_features(x, max_magnitude=300):
    for ind in range(len(x)):
        for freq in range(len(x[ind])):
            for time in range(len(x[ind][freq])):
                for feature in range(len(x[ind][freq][time])):
                    if(x[ind][freq][time][feature] <= max_magnitude):
                        x[ind][freq][time][feature] /= max_magnitude
                    else:
                        x[ind][freq][time][feature] = 1.0


current_model_number = 0


def main():

    kernel_interval = (3,4,5,6,7,8,9,10,11,12)
    vector_dimension = (2,4,6,8,10,12,14)

    hyperparameters = [{'name': 'learning_rate', 'type': 'continuous',
                        'domain': (10e-6, 10e-3)},
                       {'name': 'kernel_size_taxis', 'type': 'discrete',
                        'domain': kernel_interval},
                        {'name': 'kernel_size_faxis', 'type': 'discrete',
                        'domain': kernel_interval},
                        {'name': 'final_dimension', 'type': 'discrete',
                        'domain': vector_dimension},
                       {'name': 'regularization', 'type': 'continuous',
                        'domain': (0.001, 0.1)},
                        {'name': 'margin', 'type': 'continuous',
                        'domain': (1.0, 2.0)},
                        {'name': 'norm_factor', 'type': 'continuous',
                        'domain': (100.0, 500.0)}]


    seed = 15
    def bayesian_optimization_function(x):
        current_learning_rate = float(x[:, 0])
        current_kernel_size_taxis = int(x[:, 1])
        current_kernel_size_faxis = int(x[:, 2])
        current_final_dimension = int(x[:, 3])
        current_regularization = float(x[:, 4])
        current_margin = float(x[:, 5])
        current_norm_factor = float(x[:, 6])

        model_name = './models_optimization_logs/siamese_net_lr_' \
        + str(current_learning_rate) \
        + '_' + str(current_kernel_size_taxis) \
        + '_' + str(current_kernel_size_faxis) \
        + '_' + str(current_final_dimension) \
        + '_' + str(current_regularization) \
        + '_' + str(current_margin) \
        + '_' + str(current_norm_factor)

        print(model_name)

        global current_model_number
        current_model_number += 1

        n_splits = 5
        fold = StratifiedKFold(n_splits=n_splits, random_state=seed)

        hc = eeg.get_hc_instances()

        scz = eeg.get_scz_instances()

        population = np.append(hc, scz, axis=0)

        X = [0]*84
        ids = []
        for i in range(84):
            ids += [i]

        strat = [0]*39 + [1]*45

        cross_validation_accuracy = 0

        for train_index, test_index in fold.split(X, strat):
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(ids)[train_index], np.array(ids)[test_index]

            #define a random seed to partition the data 
            tr_pairs, tr_y, te_pairs, te_y, input_shape = data(population, y_train, 
                y_test, current_norm_factor)

            K.clear_session()
            s = siamese(input_shape, learning_rate=current_learning_rate,
                final_dimension=current_final_dimension,
                kernel_size=(current_kernel_size_taxis,current_kernel_size_faxis),
                regularization=current_regularization,
                margin=current_margin)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            with session:
                session.run(tf.global_variables_initializer())
                history = s.model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                          batch_size=number_channels*16,
                          epochs=epochs)
                y_pred = s.model.predict([te_pairs[:, 0], te_pairs[:, 1]])
                cross_validation_accuracy += compute_accuracy(te_y, y_pred)

            K.clear_session()

        cross_validation_accuracy /= n_splits
        print("Model: " + model_name +
              ' | Accuracy: ' + str(cross_validation_accuracy))

        
        print("FINISHED ITERATION\n\n\n\n\n\n\n")
        return 1 - cross_validation_accuracy

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=bayesian_optimization_function, domain=hyperparameters)

    optimizer.run_optimization(max_iter=250, verbosity=True)

    print("Values for the model should be: ")
    print("optimized parameters: {0}".format(optimizer.x_opt))
    print("[learning_rate, kernel_size_taxis, kernel_size_faxis, final_dimension, regularization, norm_factor")
    print("optimized eval_accuracy: {0}".format(1 - optimizer.fx_opt))

if __name__ == "__main__":
    main()