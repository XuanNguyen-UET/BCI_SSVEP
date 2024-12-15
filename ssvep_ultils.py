# %% [code]
# %% [markdown]
# # Utilities for CNN based SSVEP Classification

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:30:44.558322Z","iopub.execute_input":"2024-12-13T02:30:44.558731Z","iopub.status.idle":"2024-12-13T02:30:58.940337Z","shell.execute_reply.started":"2024-12-13T02:30:44.558693Z","shell.execute_reply":"2024-12-13T02:30:58.938807Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:30:58.942343Z","iopub.execute_input":"2024-12-13T02:30:58.942718Z","iopub.status.idle":"2024-12-13T02:30:59.363883Z","shell.execute_reply.started":"2024-12-13T02:30:58.942676Z","shell.execute_reply":"2024-12-13T02:30:59.362503Z"},"jupyter":{"source_hidden":true}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:33:23.200541Z","iopub.execute_input":"2024-12-13T02:33:23.201005Z","iopub.status.idle":"2024-12-13T02:33:23.291462Z","shell.execute_reply.started":"2024-12-13T02:33:23.200937Z","shell.execute_reply":"2024-12-13T02:33:23.290257Z"}}
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.signal import butter, filtfilt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers, regularizers


# %% [markdown]
# ## 1. Banpass filter for input Data 

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:39:56.967186Z","iopub.execute_input":"2024-12-13T02:39:56.967936Z","iopub.status.idle":"2024-12-13T02:39:56.976118Z","shell.execute_reply.started":"2024-12-13T02:39:56.967873Z","shell.execute_reply":"2024-12-13T02:39:56.974854Z"}}
def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    '''
    Returns bandpass filtered data between the frequency ranges specified in the input.

    Args:
        data (numpy.ndarray): array of samples. 
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        sample_rate (float): sampling rate (Hz).
        order (int): order of the bandpass filter.

    Returns:
        (numpy.ndarray): bandpass filtered data.
    '''
    
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# %% [markdown]
# ## 2. Butterworth bandpass for EEG signal

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:40:13.574220Z","iopub.execute_input":"2024-12-13T02:40:13.574625Z","iopub.status.idle":"2024-12-13T02:40:13.584172Z","shell.execute_reply.started":"2024-12-13T02:40:13.574587Z","shell.execute_reply":"2024-12-13T02:40:13.583036Z"}}
def get_filtered_eeg(eeg, lowcut, highcut, order, sample_rate):
    '''
    Returns bandpass filtered eeg for all channels and trials.

    Args:
        eeg (numpy.ndarray): raw eeg data of shape (num_classes, num_channels, num_samples, num_trials).
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        order (int): order of the bandpass filter.
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): bandpass filtered eeg of shape (num_classes, num_channels, num_samples, num_trials).
    '''
    
    num_classes = eeg.shape[0]
    num_chan = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]
    
    trial_len = int(38+0.135*sample_rate+4*sample_rate-1) - int(38+0.135*sample_rate)
    filtered_data = np.zeros((eeg.shape[0], eeg.shape[1], trial_len, eeg.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                signal_to_filter = np.squeeze(eeg[target, channel, int(38+0.135*sample_rate):
                                               int(38+0.135*sample_rate+4*sample_rate-1), 
                                               trial])
                filtered_data[target, channel, :, trial] = butter_bandpass_filter(signal_to_filter, lowcut, 
                                                                                  highcut, sample_rate, order)
    return filtered_data

# %% [markdown]
# ## 3. Data segmentation base on windows and overlap

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:43:19.061601Z","iopub.execute_input":"2024-12-13T02:43:19.062052Z","iopub.status.idle":"2024-12-13T02:43:19.069907Z","shell.execute_reply.started":"2024-12-13T02:43:19.062010Z","shell.execute_reply":"2024-12-13T02:43:19.068490Z"}}
def buffer(data, duration, data_overlap):
    '''
    Returns segmented data based on the provided input window duration and overlap.

    Args:
        data (numpy.ndarray): array of samples. 
        duration (int): window length (number of samples).
        data_overlap (int): number of samples of overlap.

    Returns:
        (numpy.ndarray): segmented data of shape (number_of_segments, duration).
    '''
    
    number_segments = int(math.ceil((len(data) - data_overlap)/(duration - data_overlap)))
    temp_buf = [data[i:i+duration] for i in range(0, len(data), (duration - int(data_overlap)))]
    temp_buf[number_segments-1] = np.pad(temp_buf[number_segments-1],
                                         (0, duration-temp_buf[number_segments-1].shape[0]),
                                         'constant')
    segmented_data = np.vstack(temp_buf[0:number_segments])
    
    return segmented_data

# %% [markdown]
# ## 4. Split EEG data into epochs base on window duration and shifts

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:55:14.574452Z","iopub.execute_input":"2024-12-13T02:55:14.574847Z","iopub.status.idle":"2024-12-13T02:55:14.583359Z","shell.execute_reply.started":"2024-12-13T02:55:14.574808Z","shell.execute_reply":"2024-12-13T02:55:14.581934Z"}}
def get_segmented_epochs(data, window_len, shift_len, sample_rate):
    '''
    Returns epoched eeg data based on the window duration and step size.

    Args:
        data (numpy.ndarray): array of samples. 
        window_len (int): window length (seconds).
        shift_len (int): step size (seconds).
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): epoched eeg data of shape. 
        (num_classes, num_channels, num_trials, number_of_segments, duration).
    '''
    
    num_classes = data.shape[0]
    num_chan = data.shape[1]
    num_trials = data.shape[3]
    
    duration = int(window_len*sample_rate)
    data_overlap = (window_len - shift_len)*sample_rate
    
    number_of_segments = int(math.ceil((data.shape[2] - data_overlap)/
                                       (duration - data_overlap)))
    
    segmented_data = np.zeros((data.shape[0], data.shape[1], 
                               data.shape[3], number_of_segments, duration))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                segmented_data[target, channel, trial, :, :] = buffer(data[target, channel, :, trial], 
                                                                      duration, data_overlap) 
    
    return segmented_data

# %% [markdown]
# ## 5. Calculate magnitude spectrum of segmented data using FFT 

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T02:56:35.080120Z","iopub.execute_input":"2024-12-13T02:56:35.081072Z","iopub.status.idle":"2024-12-13T02:56:35.090418Z","shell.execute_reply.started":"2024-12-13T02:56:35.081018Z","shell.execute_reply":"2024-12-13T02:56:35.089303Z"}}
def magnitude_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns magnitude spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): magnitude spectrum features of the input EEG.
        (n_fc, num_channels, num_classes, num_trials, number_of_segments).
    '''
    
    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1

    features_data = np.zeros(((fft_index_end - fft_index_start), 
                              segmented_data.shape[1], segmented_data.shape[0], 
                              segmented_data.shape[2], segmented_data.shape[3]))
    
    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT)/fft_len
                    magnitude_spectrum = 2*np.abs(temp_FFT)
                    features_data[:, channel, target, trial, segment] = magnitude_spectrum[fft_index_start:fft_index_end,]
    
    return features_data

# %% [markdown]
# ## 6. Calculate complex spectrum features of FFTranformed data

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T03:04:38.054116Z","iopub.execute_input":"2024-12-13T03:04:38.054543Z","iopub.status.idle":"2024-12-13T03:04:38.065229Z","shell.execute_reply.started":"2024-12-13T03:04:38.054495Z","shell.execute_reply":"2024-12-13T03:04:38.063868Z"}}
def complex_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns complex spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input. The real and imaginary parts of the input
    signal are concatenated into a single feature vector.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): complex spectrum features of the input EEG.
        (2*n_fc, num_channels, num_classes, num_trials, number_of_segments)
    '''
    
    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1

    features_data = np.zeros((2*(fft_index_end - fft_index_start), 
                              segmented_data.shape[1], segmented_data.shape[0], 
                              segmented_data.shape[2], segmented_data.shape[3]))
    
    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT)/fft_len
                    real_part = np.real(temp_FFT)
                    imag_part = np.imag(temp_FFT)
                    features_data[:, channel, target, trial, segment] = np.concatenate((
                        real_part[fft_index_start:fft_index_end,], 
                        imag_part[fft_index_start:fft_index_end,]), axis=0)
    
    return features_data

# %% [markdown]
# ## 7. Define CNN architecture for SSEVEP classification

# %% [code] {"execution":{"iopub.status.busy":"2024-12-13T03:06:33.552206Z","iopub.execute_input":"2024-12-13T03:06:33.552628Z","iopub.status.idle":"2024-12-13T03:06:33.563321Z","shell.execute_reply.started":"2024-12-13T03:06:33.552592Z","shell.execute_reply":"2024-12-13T03:06:33.561845Z"}}
def CNN_model(input_shape, CNN_PARAMS):
    '''
    Returns the Concolutional Neural Network model for SSVEP classification.

    Args:
        input_shape (numpy.ndarray): shape of input training data 
        e.g. [num_training_examples, num_channels, n_fc] or [num_training_examples, num_channels, 2*n_fc].
        CNN_PARAMS (dict): dictionary of parameters used for feature extraction.        
        CNN_PARAMS['batch_size'] (int): training mini batch size.
        CNN_PARAMS['epochs'] (int): total number of training epochs/iterations.
        CNN_PARAMS['droprate'] (float): dropout ratio.
        CNN_PARAMS['learning_rate'] (float): model learning rate.
        CNN_PARAMS['lr_decay'] (float): learning rate decay ratio.
        CNN_PARAMS['l2_lambda'] (float): l2 regularization parameter.
        CNN_PARAMS['momentum'] (float): momentum term for stochastic gradient descent optimization.
        CNN_PARAMS['kernel_f'] (int): 1D kernel to operate on conv_1 layer for the SSVEP CNN. 
        CNN_PARAMS['n_ch'] (int): number of eeg channels
        CNN_PARAMS['num_classes'] (int): number of SSVEP targets/classes

    Returns:
        (keras.Sequential): CNN model.
    '''
    
    model = Sequential()
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1), 
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]), 
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, CNN_PARAMS['kernel_f']), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="valid", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    
    return model