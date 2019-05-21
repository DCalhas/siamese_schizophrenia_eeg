#!/usr/bin/env python
# coding: utf-8
# # Classification algorithms on FFT Frequency Features
from scipy.signal import stft
from scipy.fftpack import fft
from scipy.signal import get_window
from scipy.spatial.distance import cosine, chebyshev

from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import GPy
import GPyOpt

import optimize_classifier as opt_clf

import eeg
import numpy as np

hc = eeg.get_hc_instances()

scz = eeg.get_scz_instances()

population = np.append(hc, scz, axis=0)

number_channels = 16
number_individuals = len(population)

channel_nomenclature = ['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']


def fft_to_freq_bin(fft, N, fs=128):
    #at each time step binarize the frequencies freq_map[:,0]
    f = np.linspace (0, fs, N/2)
    
    #delta, theta1, theta2, alpha1, alpha2, beta1, beta2, beta3, gamma - 9 frequency bands
                                
    frequencies = [0]*9
    
    for i in range(len(fft)):
        if(f[i] <= 4):
            frequencies[0] += fft[i]
        elif(f[i] <=6):
            frequencies[1] += fft[i]
        elif(f[i] <=8):
            frequencies[2] += fft[i]
        elif(f[i] <=10):
            frequencies[3] += fft[i]
        elif(f[i] <=13):
            frequencies[4] += fft[i]
        elif(f[i] <=16):
            frequencies[5] += fft[i]
        elif(f[i] <=23):
            frequencies[6] += fft[i]
        elif(f[i] <=30):
            frequencies[7] += fft[i]
        elif(f[i] >30):
            frequencies[8] += fft[i]
    
    return frequencies


# ### Build dataset train and test set

#mutate to fft features
binarize = False
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

iteration = 1
for train_index, test_index in loo.split(X):
    print("STARTED iteration")
    import time
    start = time.time()
    _, _ = np.array(X)[train_index], np.array(X)[test_index]
    train_labels, test_labels = np.array(ids)[train_index], np.array(ids)[test_index]

    X_train = []
    for individual in train_labels:
        for channel in range(number_channels):
            N = int(len(population[individual][:,channel])/2)
            fft_instance = fft(population[individual][:,channel])
            if(binarize):
                X_train += [fft_to_freq_bin(abs(fft_instance[range(int(N/2))]), N)]
            else:
                X_train += [abs(fft_instance[range(int(N/2))])]
            

    y_train = []
    for i in range(len(train_labels)):
        if(train_labels[i] >=39):
            y_train += [1]*16                        
        else:
            y_train += [0]*16

    X_test = []
    for individual in test_labels:
        for channel in range(number_channels):
            N = int(len(population[individual][:,channel])/2)
            fft_instance = fft(population[individual][:,channel])
            if(binarize):
                X_test += [fft_to_freq_bin(abs(fft_instance[range(int(N/2))]), N)]
            else:
                X_test += [abs(fft_instance[range(int(N/2))])]

    y_test = []
    for i in range(len(test_labels)):
        if(test_labels[i] >=39):
            y_test += [1]*16
        else:
            y_test += [0]*16
    
    #fit svm to data
    #######################################
    #SUPPORT VECTOS MACHINE
    #######################################

    #optimize hyperparameters for SVM in this partition
    hyperparameters_svc = [{'name': 'cost', 'type': 'continuous',
                        'domain': (0.5, 5.0)},
                        {'name': 'kernel', 'type': 'discrete',
                        'domain': (0, 1)}]#0-linear, 1-rbf
    opt_hyperparameters_svc = opt_clf.optimize_classifier(X_train, y_train, SVC, hyperparameters_svc)
    if(opt_hyperparameters_svc[1]):
        svm = SVC(C=opt_hyperparameters_svc[0], kernel='rbf')
    else:
        svm = SVC(C=opt_hyperparameters_svc[0], kernel='linear')
    #train with the hyperparameters given
    svm.fit(X_train, y_train)

    #######################################
    #NAIVE BAYES
    #######################################

    #nothing to optimize

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    #######################################
    #K NEAREST NEIGHBORS
    #######################################

    #optimize k value
    hyperparameters_knn = [{'name': 'k', 'type': 'discrete',
                        'domain': (2, 3, 4, 5, 6, 7, 8)}]
    opt_hyperparameters_knn = opt_clf.optimize_classifier(X_train, y_train, KNeighborsClassifier, hyperparameters_knn)

    knn = KNeighborsClassifier(n_neighbors=int(opt_hyperparameters_knn[0]))
    knn.fit(X_train, y_train)
    
    #######################################
    #RANDOM FOREST
    #######################################

    hyperparameters_rf = [{'name': 'n_estimators', 'type': 'discrete',
                        'domain': (5, 10, 15, 20, 25)}]
    opt_hyperparameters_rf = opt_clf.optimize_classifier(X_train, y_train, RandomForestClassifier, hyperparameters_rf)

    rf = RandomForestClassifier(n_estimators=int(opt_hyperparameters_rf[0]))
    rf.fit(X_train, y_train)
    
    #######################################
    #XGBOOST
    #######################################

    hyperparameters_xgb = [{'name': 'max_depth', 'type': 'discrete',
                        'domain': (3, 4, 5, 6, 7)},
                        {'name': 'learning_rate', 'type': 'continuous',
                        'domain': (0.001, 0.1)},
                        {'name': 'n_estimators', 'type': 'discrete',
                        'domain': (10, 50, 100, 200)}]
    opt_hyperparameters_xgb = opt_clf.optimize_classifier(X_train, y_train, xgb.XGBClassifier, hyperparameters_xgb)

    xgbclf = xgb.XGBClassifier(max_depth=int(opt_hyperparameters_xgb[0]),
        learning_rate=float(opt_hyperparameters_xgb[1]),
        n_estimators=int(opt_hyperparameters_xgb[2]))
    xgbclf.fit(np.array(X_train), np.array(y_train))

    ##############################################################################
    ##############################################################################
    ##############################################################################

    #get the result
    #SUPPORT VECTOS MACHINE
    predictions = list(svm.predict(X_test))
    classifiers['Support Vector Machine'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['Support Vector Machine'] += [predictions[channel]]
    #NAIVE BAYES
    predictions = list(nb.predict(X_test))
    classifiers['Naive Bayes'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['Naive Bayes'] += [predictions[channel]]
    #K NEAREST NEIGHBORS
    predictions = list(knn.predict(X_test))
    classifiers['K-Nearest Neighbors'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['K-Nearest Neighbors'] += [predictions[channel]]
    #RANDOM FOREST
    predictions = list(rf.predict(X_test))
    classifiers['Random Forest'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['Random Forest'] += [predictions[channel]]
    #XGBOOST
    predictions = list(xgbclf.predict(X_test))
    classifiers['XGBoost'] += predictions
    for channel in range(len(predictions)):
        classifiers_by_channel[channel]['XGBoost'] += [predictions[channel]]

    print("This iteration took around: ", time.time()-start, " seconds")
    print("Finished iteration: ", iteration, end="\n\n\n\n\n\n\n\n\n\n")
    iteration += 1


# ### Compute Accuracy
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