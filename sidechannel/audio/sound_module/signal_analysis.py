#!/usr/bin/env python3.8
# coding: utf-8

import sys
import librosa
from scipy.signal import butter, sosfilt, sosfreqz
import numpy as np
from numpy.lib.stride_tricks import as_strided

###############################################################################
######                            FREQUENCY LIMITATION                   ######
############################################################################### 

def butter_bandpass(lowcut, highcut, fs, order=9):
    ### Butterfilter from https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/12233959#12233959
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def limit_frequencies(samples, sampling_rate, lower_end, upper_end):
    """Reduces the frequency band of a signal by applying a chosen filter and returns the filtered signal
    Input:
        samples: signal data from librosa library
        sampling rate: sampling rate of the signal
        lower_end: lowest frequency for filtering cutoff
        upper_end: highest frequency for filtering cutoff
    Output:
        filtered signal"""
    filtered_samples = butter_bandpass_filter(samples, lower_end,upper_end, sampling_rate)
    return filtered_samples

###############################################################################
######                            FEATURE EXTRACTION                     ######
############################################################################### 

def get_silent_action_start(sample, sampling_rate):
    """Extracts the beginning of an action in a sample with silent background"""
    # to adjust finding, play around with top_db
    parts = librosa.effects.split(sample, top_db=50, ref=np.max, frame_length=1024, hop_length=256)
    # start is in first part:
    start_frame = parts[0][0]
    start_time = parts[0][0]/sampling_rate
    return (start_frame, start_time)

def rms_energy(samples, sampling_rate):
    """Calculates the rms of a signal and returns the rms and times
    returns a tuple of 
        S: spectrogram magnitude
        rms: root mean square of input signl [[rms_values]]
        times: timestamps of rms"""
    S, phase = librosa.magphase(librosa.stft(samples))
    rms = librosa.feature.rms(S=S)
    times = librosa.times_like(rms)
    return (S, rms, times)

def rms_energy_smoothed(rms, times, n=30):
    """Smoothes the rms energy of a signal by applying a smoothing function, n indicates how many neighbors are considered
    for the smoothing
    Input: 
        rms: rms energy of a signl
        times: timestamps of RMS energy values
    Output:
        tuple (avg_rms, avg_times)
        avg_rms: average rms depending on considered neighbors np.ndarray [shape=(t,)]
        avg_tims: timestamps of average_rms values """
    averaged_rms = moving_average(rms[0], N=n)
    avg_times = moving_average(times, N=n)
    return (averaged_rms, avg_times)


def get_event_start(rms, rms_times, thresh=0.001, w=60, start=0):
    """Get action start from rms of a signal, the start is determined based a threshold
    that needs to be exceeded in a certain time window
    Input:
        rms: rms of signal
        rms_times: timestamps of rms
    Output:
        timestamp of identified start or (-1) if thresh was not exceeded"""
    diffs = running_max_min(rms, w, 1)
    for i, j in enumerate(diffs):
            if j > thresh and i > start:
                # return first occurence
                return(i, rms_times[i])
    return (-1, -1)
    
def get_event_end(rms, rms_times, action_start_index, thresh=0.0004, w=50):
    """Returns index and end time of autmatically detected event, threshold based"""
    diffs = running_max_min(rms, w, 1)
    for i, j in enumerate(diffs):
        if i > action_start_index and j < thresh:
            return (i, rms_times[min(i+w-1, len(rms_times)-1)]) #max windows to the end #BUG: this function for some reason does not behave deterministically
    return (-1,-1)

def detect_events(rms, rms_times):
    """TODO Description: in general: return start and end-time of an event"""
    # TODO: beep detection improvement/refactor
    events = [(0,0,0,0)]
    for i in range(20):        
        (start,start_time) = get_event_start(rms, rms_times,w=8,thresh=0.0007,start=events[-1][2])
        (ending, end_time) = get_event_end(rms, rms_times, start, w=8,thresh=0.0006)
        if start == -1:
            break
        events.append((start, start_time, ending, end_time))
    # for error beeps merge two events
    beeps = []
    for i in range(1, len(events)-1,2):
        beeps.append((events[i][0],events[i][1],events[i+1][2],events[i+1][3]))
    #return events[1:]
    #print(events)
    #print(beeps)
    return beeps

###############################################################################
######                            SMOOTHING                              ######
############################################################################### 

def moving_average(a, N=3):
    """Smoothes values by calculating the mooving average between neighbors
    Input: 
        array a: values that should be smoothed
        N: amount of neighbors used to calculate the average (default 3)
    Output:
        array with smoothed values"""
    #https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    cumsum = np.cumsum(np.insert(a, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

###############################################################################
######                            MISC HELPERS                     ######
############################################################################### 

def running_max_min(a, window_size, step_size):
    # https://stackoverflow.com/questions/51808573/running-window-of-max-min-in-a-numpy-array
    nrows = (a.size - window_size)//step_size + 1
    n = a.strides[0]
    s = as_strided(a, shape=(nrows, window_size), strides=(step_size*n, n))
    return s.ptp(1)