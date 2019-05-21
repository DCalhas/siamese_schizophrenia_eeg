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

import tensorflow as tf

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
import time as time_lib

import os

import eeg

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
    def __init__(self, reg=0.011, kernel_size=(6,6), lr= 0.0004, final_dim=12):
        self.cnn = create_base_network(input_shape, kernel_size, reg, final_dim)

        adam = Adam(lr=lr)
        self.cnn.compile(loss='mean_squared_error', optimizer=adam, metrics=['binary_accuracy'])

hc = eeg.get_hc_instances()

scz = eeg.get_scz_instances()

population = np.append(hc, scz, axis=0)


# the data, split between train and test sets
binarize=False
nperseg = 512

X = [0]*84
ids = []
for i in range(84):
    ids += [i]

strat = [0]*39 + [1]*45


loo = LeaveOneOut()

n_partition = 0

channel_predictions = []
for i in range(number_channels):
    channel_predictions += [[]]
mixed_predictions = []

#change these values according to the ones obtained by script snn_hyperparameter_optimization.py
#the ones present here were obtained by a past execution
normalization_factor = 1.29816006e+02
regularization = 8.39240857e-02
final_dimension = 2
kernel_size = (4,7)
learning_rate = 9.18306459e-03


for train_index, test_index in loo.split(X):
    start = time_lib.time()

    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(ids)[train_index], np.array(ids)[test_index]


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

    print("Normalizing features")
    normalize_features(X_train, max_magnitude=normalization_factor)
    print("Finished train features")
    normalize_features(X_test, max_magnitude=normalization_factor)
    print("Finished test features")

    K.clear_session()

    cnn = convolutional(reg=regularization, kernel_size=kernel_size, lr=learning_rate, final_dim=final_dimension)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    with session:
        session.run(tf.global_variables_initializer())
        #if model has a good validation decrease, model checkpoint can be
        #removed
        history = cnn.cnn.fit(X_train, y_train,
                  batch_size=number_channels,
                  epochs=epochs)

        predictions = cnn.cnn.predict(X_test)

        if(np.mean(predictions) >= 0.5):
            mixed_predictions += [1]
        else:
            mixed_predictions += [0]

        for channel in range(len(cnn.cnn.predict(X_test))):
            channel_predictions[channel] += [predictions[channel][0]]

    K.clear_session()


    print("============================================")
    print("ENDED PARTITION ", n_partition, " taking a total of ", time_lib.time()-start, "seconds")
    print("============================================")
    n_partition += 1

############################################################################
#
#                          CLASSIFICATION REPORT
#
############################################################################

y = []
for i in range(84):
    if(i < 39):
        y += [0]
    else:
        y += [1]


#########################################################################################
#
#                               ACCURACY COMPUTATION
#
#########################################################################################
#compute mean accuracy
accuracy_mean = 0
for prediction in range(len(mixed_predictions)):
    if(mixed_predictions[prediction] == y[prediction]):
        accuracy_mean += 1
accuracy_mean /= len(y)

#compute standard deviation accuracy
accuracy_std = 0
for prediction in range(len(mixed_predictions)):
    if(mixed_predictions[prediction] == y[prediction]):
        accuracy_std += (1 - accuracy_mean)**2
accuracy_std = (accuracy_std/len(y))**(1/2)


print("Leave One Out Cross Validation Accuracy of CNN is: ", accuracy_mean, "\pm", accuracy_std)

for channel in range(number_channels):
    #compute mean accuracy
    accuracy_mean = 0
    for prediction in range(len(channel_predictions[channel])):
        if(channel_predictions[channel][prediction] == y[prediction]):
            accuracy_mean += 1
    accuracy_mean /= len(y)

    #compute standard deviation accuracy
    accuracy_std = 0
    for prediction in range(len(channel_predictions[channel])):
        if(channel_predictions[channel][prediction] == y[prediction]):
            accuracy_std += (1 - accuracy_mean)**2
    accuracy_std = (accuracy_std/len(y))**(1/2)


    print("Leave One Out Cross Validation Accuracy in channel ", channel_nomenclature[channel],
        "of CNN trained in single channel is: ", accuracy_mean, "\pm", accuracy_std)


#########################################################################################
#
#                               SENSITIVITY COMPUTATION
#
#########################################################################################

#compute mean sensitivity
sensitivity_mean = 0
positives = 0
for prediction in range(len(mixed_predictions)):
    if(y[prediction] == 1):
        positives += 1
        if(mixed_predictions[prediction] == y[prediction]):
            sensitivity_mean += 1
sensitivity_mean /= positives

#compute standard deviation sensitivity
sensitivity_std = 0
for prediction in range(len(mixed_predictions)):
    if(y[prediction] == 1 and mixed_predictions[prediction] == y[prediction]):
        sensitivity_std += (1 - sensitivity_mean)**2
sensitivity_std = (sensitivity_std/positives)**(1/2)


print("Leave One Out Cross Validation Sensitivity of CNN is: ", sensitivity_mean, "\pm", sensitivity_std)

for channel in range(number_channels):
    #compute mean sensitivity
    sensitivity_mean = 0
    positives = 0
    for prediction in range(len(channel_predictions[channel])):
        if(y[prediction] == 1):
            positives += 1
            if(channel_predictions[channel][prediction] == y[prediction]):
                sensitivity_mean += 1
    sensitivity_mean /= positives

    #compute standard deviation sensitivity
    sensitivity_std = 0
    for prediction in range(len(channel_predictions[channel])):
        if(y[prediction] == 1 and channel_predictions[channel][prediction] == y[prediction]):
            sensitivity_std += (1 - sensitivity_mean)**2
    sensitivity_std = (sensitivity_std/positives)**(1/2)


    print("Leave One Out Cross Validation Sensitivity in channel ", channel_nomenclature[channel],
        "of CNN trained in single channel is: ", sensitivity_mean, "\pm", sensitivity_std)


#########################################################################################
#
#                               SPECIFICITY COMPUTATION
#
#########################################################################################

#compute mean accuracy
specificity_mean = 0
negatives = 0
for prediction in range(len(mixed_predictions)):
    if(y[prediction] == 0):
        negatives += 1
        if(mixed_predictions[prediction] == y[prediction]):
            specificity_mean += 1
specificity_mean /= negatives

#compute standard deviation accuracy
specificity_std = 0
for prediction in range(len(mixed_predictions)):
    if(y[prediction] == 0 and mixed_predictions[prediction] == y[prediction]):
        specificity_std += (1 - specificity_mean)**2
specificity_std = (specificity_std/negatives)**(1/2)


print("Leave One Out Cross Validation Specificity of CNN is: ", specificity_mean, "\pm", specificity_std)

for channel in range(number_channels):
    #compute mean accuracy
    specificity_mean = 0
    negatives = 0
    for prediction in range(len(channel_predictions[channel])):
        if(y[prediction] == 0):
            negatives += 1
            if(channel_predictions[channel][prediction] == y[prediction]):
                specificity_mean += 1
    specificity_mean /= negatives

    #compute standard deviation accuracy
    specificity_std = 0
    for prediction in range(len(channel_predictions[channel])):
        if(y[prediction] == 0 and channel_predictions[channel][prediction] == y[prediction]):
            specificity_std += (1 - specificity_mean)**2
    specificity_std = (specificity_std/negatives)**(1/2)

    print("Leave One Out Cross Validation Specificity in channel ", channel_nomenclature[channel],
        "of CNN trained in single channel is: ", specificity_mean, "\pm", specificity_std)
