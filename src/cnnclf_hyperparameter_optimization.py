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
from keras.metrics import binary_accuracy

import tensorflow as tf

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn import metrics
import time as time_lib

import os

import eeg

import GPy
import GPyOpt

from scipy.signal import stft
from scipy.signal import get_window
from scipy.spatial.distance import cosine, chebyshev

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

num_classes = 2
epochs = 20
number_channels = 16

frequencies_nomenclature = ['delta', 'theta1', 'theta2', 'alpha1', 'alpha2', 
                   'beta1', 'beta2', 'beta3', 'gamma']

channel_nomenclature = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
                    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']


binarize = False
nperseg = 512

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

#gather a channel to train a network
def get_channel(x, y, channel, n_channels=16):
    X_channel = []
    y_channel = []

    for instance in range(channel, len(x), n_channels):
        X_channel += [x[instance]]
        y_channel += [y[instance]]

    return np.array(X_channel), y_channel

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

def create_base_network(input_shape, kernel_size, reg, final_dim):
    ##model building
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    #flatten since too many dimensions, we only want a classification output
    model.add(Conv2D(1, kernel_size=kernel_size,
                     activation='relu',
                     input_shape=input_shape, kernel_initializer=init,
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l1(reg),#0.011
                bias_regularizer=regularizers.l1(reg)))#0.011
    if(reg > 0.0):
        model.add(Dropout(0.5))
    model.add(Conv2D(1, kernel_size=kernel_size,
                    activation='relu', kernel_initializer=init,
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l1(reg),#0.011
                bias_regularizer=regularizers.l1(reg)))#0.011
    if(reg > 0.0):
        model.add(Dropout(0.5))
    #things to test in order to increase the performance of the mdel
    #play a little with the kernel sizes - test values: (6,6)
    #change the optimization function - 
    model.add(Flatten())
    #embedding sizes with better results seem to be between [8,15[
    model.add(Dense(final_dim, activation='softmax', kernel_initializer=init,
                bias_initializer='zeros'))#13
    model.add(Dense(1, activation='relu',kernel_initializer=init,
                bias_initializer='zeros'))
    print(model.summary())
    return model

class convolutional:
    def __init__(self, input_shape, reg=0.011, kernel_size=(6,6), lr= 0.0004, final_dim=12):
        self.cnn = create_base_network(input_shape, kernel_size, reg, final_dim)

        adam = Adam(lr=lr)
        self.cnn.compile(loss='mean_squared_error', optimizer=adam, metrics=['binary_accuracy'])

def compute_accuracy(y_pred, y_true):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def data(population, ids, norm_factor):
    X = np.array(get_population_stft_by_bins(population, ids, binarize=binarize, 
    n_channels=number_channels, nperseg=nperseg))

    y = []
    for i in range(len(ids)):
        if(ids[i] >= 39):
            y += [1]*number_channels
        else:
            y += [0]*number_channels

    time = len(X[0])
    freq = len(X[0][0])
    X = X.reshape(X.shape[0], time, freq, 1)
    input_shape = (time, freq, 1)
    #more reshaping
    X = X.astype('float32')

    print("Normalizing ")
    normalize_features(X, max_magnitude=norm_factor)

    return X, y, input_shape


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
                        {'name': 'norm_factor', 'type': 'continuous',
                        'domain': (100.0, 500.0)}]


    seed = 15
    def bayesian_optimization_function(x):
        start = time_lib.time()

        current_learning_rate = float(x[:, 0])
        current_kernel_size_taxis = int(x[:, 1])
        current_kernel_size_faxis = int(x[:, 2])
        current_final_dimension = int(x[:, 3])
        current_regularization = float(x[:, 4])
        current_norm_factor = float(x[:, 5])

        model_name = './models_optimization_logs/cnnclf_net_lr_' \
        + str(current_learning_rate) \
        + '_' + str(current_kernel_size_taxis) \
        + '_' + str(current_kernel_size_faxis) \
        + '_' + str(current_final_dimension) \
        + '_' + str(current_regularization) \
        + '_' + str(current_norm_factor)

        print(model_name)

        global current_model_number
        current_model_number += 1


        n_splits = 5
        fold = StratifiedKFold(n_splits=n_splits, random_state=seed)

        hc = eeg.get_hc_instances()

        scz = eeg.get_scz_instances()

        population = np.append(hc, scz, axis=0)

        X_pop = [0]*84
        ids = []
        for i in range(84):
            ids += [i]

        strat = [0]*39 + [1]*45

        cross_validation_accuracy = 0

        #normalization is performed here
        X, y, input_shape = data(population, ids,
                current_norm_factor)

        for train_index, test_index in fold.split(X_pop, strat):
            new_train_index = []
            for i in range(len(train_index)):
                for j in range(train_index[i]*number_channels, train_index[i]*number_channels+number_channels):
                    new_train_index += [j]

            train_index = np.array(new_train_index)

            new_test_index = []
            for i in range(len(test_index)):
                for j in range(test_index[i]*number_channels, test_index[i]*number_channels+number_channels):
                    new_test_index += [j]
            test_index = np.array(new_test_index)

            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

            K.clear_session()

            cnn = convolutional(input_shape, reg=current_regularization, 
                kernel_size=(current_kernel_size_taxis, current_kernel_size_faxis), 
                lr=current_learning_rate, final_dim=current_final_dimension)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            with session:
                session.run(tf.global_variables_initializer())
                history = cnn.cnn.fit(X_train, y_train,
                          batch_size=number_channels,
                          epochs=epochs)

                y_pred = cnn.cnn.predict(X_test)

                accuracy_mean = 0
                for prediction in range(len(y_pred)):
                    if(y_pred[prediction] == y_test[prediction]):
                        accuracy_mean += 1
                accuracy_mean /= len(y_test)

                print(accuracy_mean)

                cross_validation_accuracy += accuracy_mean

            K.clear_session()

        cross_validation_accuracy /= n_splits
        print("Model: " + model_name +
              ' | Accuracy: ' + str(cross_validation_accuracy))

        print("Took a total of  ", time_lib.time()-start, " seconds")
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