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

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


import tensorflow as tf

import os

import optimize_classifier as opt_clf

import time as time_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

num_classes = 2
epochs = 20
number_channels = 16

number_individuals = 84

from scipy.signal import stft
from scipy.signal import get_window
from scipy.spatial.distance import cosine, chebyshev

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
        (self.base_network).save_weights("./validation_models/base_network_partition_" + str(k) + ".h5")

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

import eeg
import numpy as np

hc = eeg.get_hc_instances()

scz = eeg.get_scz_instances()

population = np.append(hc, scz, axis=0)


from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
# the data, split between train and test sets
binarize=False
nperseg = 512

X = [0]*84
ids = []
for i in range(84):
    ids += [i]

strat = [0]*39 + [1]*45


loo = LeaveOneOut()

classifiers = {'Support Vector Machine': [], 'Naive Bayes': [], 
               'K-Nearest Neighbors':[], 'Random Forest': [], 'XGBoost': []}


classifiers_by_channel = []
for i in range(number_channels):
    classifiers_by_channel += [{'Support Vector Machine': [], 'Naive Bayes': [], 
               'K-Nearest Neighbors':[], 'Random Forest': [], 'XGBoost': []}]


n_partition = 0

#change these values according to the ones obtained by script snn_hyperparameter_optimization.py
#the ones present here were obtained by a past execution
normalization_factor = 1.04842448e+02
margin_factor = 1.18389688e+00
regularization = 1.65531895e-02
final_dimension = 10
kernel_size = (5, 11)
learning_rate = 1.49586563e-03

for train_index, test_index in loo.split(X):
    start = time_lib.time()
    
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(ids)[train_index], np.array(ids)[test_index]

    print(y_train, y_test)
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

    normalize_features(X_train, max_magnitude=normalization_factor)
    normalize_features(X_test, max_magnitude=normalization_factor)

    #need to add 1 dimension to tuple because convolution requires it to
    print(input_shape)

    # create training+test positive and negative pairs
    tr_pairs, tr_y = create_pairs(X_train, y_train, n_channels=number_channels)

    K.clear_session()

    s = siamese(input_shape, learning_rate=learning_rate,
                final_dimension=final_dimension,
                kernel_size=kernel_size,
                regularization=regularization,
                margin=margin_factor)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    with session:
        session.run(tf.global_variables_initializer())
        #if model has a good validation decrease, model checkpoint can be
        #removed

        print("============================================")
        print("STARTING PARTITION ", n_partition)
        print("============================================")

        history = s.model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=number_channels*16,
                  epochs=epochs)

        y_pred = s.model.predict([tr_pairs[:, 0], 
                                tr_pairs[:, 1]])
        tr_acc = compute_accuracy(tr_y, y_pred)

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

        s.save_base_network(n_partition)

        #extract the embeddings from the base network
        X_train_features = s.base_network.predict(X_train)
        X_test_features = s.base_network.predict(X_test)

    K.clear_session()
    #fit svm to data
    #######################################
    #SUPPORT VECTOS MACHINE
    #######################################

    #optimize hyperparameters for SVM in this partition
    hyperparameters_svc = [{'name': 'cost', 'type': 'continuous',
                        'domain': (0.5, 5.0)},
                        {'name': 'kernel', 'type': 'discrete',
                        'domain': (0, 1)}]#0-linear, 1-rbf
    opt_hyperparameters_svc = opt_clf.optimize_classifier(X_train_features, y_train, SVC, hyperparameters_svc)
    if(opt_hyperparameters_svc[1]):
        svm = SVC(C=opt_hyperparameters_svc[0], kernel='rbf')
    else:
        svm = SVC(C=opt_hyperparameters_svc[0], kernel='linear')
    #train with the hyperparameters given
    svm.fit(X_train_features, y_train)

    #######################################
    #NAIVE BAYES
    #######################################

    #nothing to optimize

    nb = GaussianNB()
    nb.fit(X_train_features, y_train)

    #######################################
    #K NEAREST NEIGHBORS
    #######################################

    #optimize k value
    hyperparameters_knn = [{'name': 'k', 'type': 'discrete',
                        'domain': (2, 3, 4, 5, 6, 7, 8)}]
    opt_hyperparameters_knn = opt_clf.optimize_classifier(X_train_features, y_train, KNeighborsClassifier, hyperparameters_knn)

    knn = KNeighborsClassifier(n_neighbors=int(opt_hyperparameters_knn[0]))
    knn.fit(X_train_features, y_train)
    
    #######################################
    #RANDOM FOREST
    #######################################

    hyperparameters_rf = [{'name': 'n_estimators', 'type': 'discrete',
                        'domain': (5, 10, 15, 20, 25)}]
    opt_hyperparameters_rf = opt_clf.optimize_classifier(X_train_features, y_train, RandomForestClassifier, hyperparameters_rf)

    rf = RandomForestClassifier(n_estimators=int(opt_hyperparameters_rf[0]))
    rf.fit(X_train_features, y_train)
    
    #######################################
    #XGBOOST
    #######################################

    hyperparameters_xgb = [{'name': 'max_depth', 'type': 'discrete',
                        'domain': (3, 4, 5, 6, 7)},
                        {'name': 'learning_rate', 'type': 'continuous',
                        'domain': (0.001, 0.1)},
                        {'name': 'n_estimators', 'type': 'discrete',
                        'domain': (10, 50, 100, 200)}]
    opt_hyperparameters_xgb = opt_clf.optimize_classifier(X_train_features, y_train, xgb.XGBClassifier, hyperparameters_xgb)

    xgbclf = xgb.XGBClassifier(max_depth=int(opt_hyperparameters_xgb[0]),
        learning_rate=float(opt_hyperparameters_xgb[1]),
        n_estimators=int(opt_hyperparameters_xgb[2]))
    xgbclf.fit(np.array(X_train_features), np.array(y_train))

    ##############################################################################
    ##############################################################################
    ##############################################################################

    #get the result
    #SUPPORT VECTOS MACHINE
    predictions = list(svm.predict(X_test_features))
    classifiers['Support Vector Machine'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['Support Vector Machine'] += [predictions[channel]]
    #NAIVE BAYES
    predictions = list(nb.predict(X_test_features))
    classifiers['Naive Bayes'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['Naive Bayes'] += [predictions[channel]]
    #K NEAREST NEIGHBORS
    predictions = list(knn.predict(X_test_features))
    classifiers['K-Nearest Neighbors'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['K-Nearest Neighbors'] += [predictions[channel]]
    #RANDOM FOREST
    predictions = list(rf.predict(X_test_features))
    classifiers['Random Forest'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['Random Forest'] += [predictions[channel]]
    #XGBOOST
    predictions = list(xgbclf.predict(X_test_features))
    classifiers['XGBoost'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['XGBoost'] += [predictions[channel]]

    print("This iteration took around: ", time_lib.time()-start, " seconds")
    print("Finished iteration: ", n_partition, end="\n\n\n\n\n\n\n\n\n\n")
    n_partition += 1

############################################################################
#
#                          CLASSIFICATION REPORT
#
############################################################################

y = []
for i in range(number_individuals):
    if(i >= 39):
        y += [1]*16
    else:
        y += [0]*16

y_channel = []

for i in range(number_individuals):
    if(i >= 39):
        y_channel += [1]
    else:
        y_channel += [0]

#########################################################################################
#
#                               ACCURACY COMPUTATION
#
#########################################################################################
for classifier in classifiers.keys():
    #compute mean accuracy
    accuracy_mean = 0
    for prediction in range(len(classifiers[classifier])):
        if(classifiers[classifier][prediction] == y[prediction]):
            accuracy_mean += 1
    accuracy_mean /= len(y)

    #compute standard deviation accuracy
    accuracy_std = 0
    for prediction in range(len(classifiers[classifier])):
        if(classifiers[classifier][prediction] == y[prediction]):
            accuracy_std += (1 - accuracy_mean)**2
    accuracy_std = (accuracy_std/len(y))**(1/2)


    print("Leave One Out Cross Validation Accuracy of ", classifier, 
        " is: ", accuracy_mean, "\pm", accuracy_std)

    for channel in range(number_channels):
        #compute mean accuracy
        accuracy_mean = 0
        for prediction in range(len(classifiers_by_channel[channel][classifier])):
            if(classifiers_by_channel[channel][classifier][prediction] == y_channel[prediction]):
                accuracy_mean += 1
        accuracy_mean /= len(y_channel)

        #compute standard deviation accuracy
        accuracy_std = 0
        for prediction in range(len(classifiers_by_channel[channel][classifier])):
            if(classifiers_by_channel[channel][classifier][prediction] == y_channel[prediction]):
                accuracy_std += (1 - accuracy_mean)**2
        accuracy_std = (accuracy_std/len(y_channel))**(1/2)


        print("Leave One Out Cross Validation Accuracy in channel ",  channel_nomenclature[channel],
            " of classifier", classifier,
            " is: ", accuracy_mean, "\pm", accuracy_std)


#########################################################################################
#
#                               SENSITIVITY COMPUTATION
#
#########################################################################################
for classifier in classifiers.keys():
    #compute mean sensitivity
    sensitivity_mean = 0
    positives = 0
    for prediction in range(len(classifiers[classifier])):
        if(y[prediction] == 1):
            positives += 1
            if(classifiers[classifier][prediction] == y[prediction]):
                sensitivity_mean += 1
    sensitivity_mean /= positives

    #compute standard deviation sensitivity
    sensitivity_std = 0
    for prediction in range(len(classifiers[classifier])):
        if(y[prediction] == 1 and classifiers[classifier][prediction] == y[prediction]):
            sensitivity_std += (1 - sensitivity_mean)**2
    sensitivity_std = (sensitivity_std/positives)**(1/2)


    print("Leave One Out Cross Validation Sensitivity of ", classifier, 
        " is: ", sensitivity_mean, "\pm", sensitivity_std)

    for channel in range(number_channels):
        #compute mean sensitivity
        sensitivity_mean = 0
        positives = 0
        for prediction in range(len(classifiers_by_channel[channel][classifier])):
            if(y_channel[prediction] == 1):
                positives += 1
                if(classifiers_by_channel[channel][classifier][prediction] == y_channel[prediction]):
                    sensitivity_mean += 1
        sensitivity_mean /= positives

        #compute standard deviation sensitivity
        sensitivity_std = 0
        for prediction in range(len(classifiers_by_channel[channel][classifier])):
            if(y_channel[prediction] == 1 and classifiers_by_channel[channel][classifier][prediction] == y_channel[prediction]):
                sensitivity_std += (1 - sensitivity_mean)**2
        sensitivity_std = (sensitivity_std/positives)**(1/2)


        print("Leave One Out Cross Validation Sensitivity in channel ",  channel_nomenclature[channel],
            " of classifier", classifier, 
            " is: ", sensitivity_mean, "\pm", sensitivity_std)

#########################################################################################
#
#                               SPECIFICITY COMPUTATION
#
#########################################################################################
for classifier in classifiers.keys():
    #compute mean accuracy
    specificity_mean = 0
    negatives = 0
    for prediction in range(len(classifiers[classifier])):
        if(y[prediction] == 0):
            negatives += 1
            if(classifiers[classifier][prediction] == y[prediction]):
                specificity_mean += 1
    specificity_mean /= negatives

    #compute standard deviation accuracy
    specificity_std = 0
    for prediction in range(len(classifiers[classifier])):
        if(y[prediction] == 0 and classifiers[classifier][prediction] == y[prediction]):
            specificity_std += (1 - specificity_mean)**2
    specificity_std = (specificity_std/negatives)**(1/2)


    print("Leave One Out Cross Validation Speficity of ", classifier, 
        " is: ", specificity_mean, "\pm", specificity_std)

    for channel in range(number_channels):
        #compute mean accuracy
        specificity_mean = 0
        negatives = 0
        for prediction in range(len(classifiers_by_channel[channel][classifier])):
            if(y_channel[prediction] == 0):
                negatives += 1
                if(classifiers_by_channel[channel][classifier][prediction] == y_channel[prediction]):
                    specificity_mean += 1
        specificity_mean /= negatives

        #compute standard deviation accuracy
        specificity_std = 0
        for prediction in range(len(classifiers_by_channel[channel][classifier])):
            if(y_channel[prediction] == 0 and classifiers_by_channel[channel][classifier][prediction] == y_channel[prediction]):
                specificity_std += (1 - specificity_mean)**2
        specificity_std = (specificity_std/negatives)**(1/2)


        print("Leave One Out Cross Validation Speficity in channel ",  channel_nomenclature[channel],
            " of classifier", classifier,
            " is: ", specificity_mean, "\pm", specificity_std)