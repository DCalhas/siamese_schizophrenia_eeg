import numpy as np


######################################################################
#
# get file names from schizophrenia group
#change the directory path to what best suits you
#
######################################################################
def get_scz_files(directory='../dataset/scz'):
    from os import listdir
    from os.path import isfile, join
    eea_files = [f for f in listdir(directory) if isfile(join(directory, f))]

    return eea_files

######################################################################
#
# get file names from healthy control group
#change the directory path to what best suits you
#
######################################################################
def get_hc_files(directory='../dataset/healthy'):
    from os import listdir
    from os.path import isfile, join
    eea_files = [f for f in listdir(directory) if isfile(join(directory, f))]

    return eea_files


######################################################################
#
# reads .eea files from the EEG dataset 
# gives format where the rows are the 7680 samples and the columns
# are the 16 channels
# so if you want a channel you just have to take the 
# column of the matrix
#
######################################################################
def read_eea(file_name, samples=7680, channels=16):
    data = []
    
    eea = open(file_name, 'r')

    lines = eea.readlines()
    
    for c in range(channels):
        channel = []
        for s in range(samples):
            channel += [float(lines[c+s][:-2])]
        data += [channel]
        
    eea.close()
    
    data = np.array(data)
    
    return np.transpose(data)


######################################################################
#
# plots the spectral density of a time series with the values from 
# scipy.signal.stft
#
######################################################################
def plot_spectral_density(t, f, Zxx):
    amplitude = f[-1]
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amplitude)

    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


######################################################################
#
# get all the EEG instances of schizophrenia patients
#
######################################################################
def get_scz_instances():
    scz_files = get_scz_files()

    #change this path to the dataset path of healthy controls
    directory = '../dataset/scz/'
    
    instances = []
    
    for scz_file in scz_files:
        instances += [read_eea(directory + scz_file)]
        
    return np.array(instances)


######################################################################
#
# get all the EEG instances of healthy controls
#
######################################################################
def get_hc_instances():
    hc_files = get_hc_files()

    #change this path to the dataset path of schizphrenian individuals
    directory = '../dataset/healthy/'
    
    instances = []
    
    for hc_file in hc_files:
        instances += [read_eea(directory + hc_file)]
        
    return np.array(instances)