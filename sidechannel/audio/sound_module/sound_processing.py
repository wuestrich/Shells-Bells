#! /usr/bin/env python3.7

# intro do sound processing: https://ia600309.us.archive.org/13/items/IntroductionToSoundProcessing/vsp.pdf 
# tut librosa https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520 

import sys
import librosa
from scipy.signal import butter, sosfilt, sosfreqz
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal.spectral import spectrogram
import scipy.optimize # pylint: disable=import-error
from scipy.optimize import OptimizeWarning # pylint: disable=import-error
from scipy.stats import wasserstein_distance # pylint: disable=import-error
import matplotlib.patches as patches

f_path = "../data/sound/"
# files: circle.wav, square.wav testbed_fan_on.wav

def read_file(path):
    samples, sampling_rate= librosa.load(path)
    return (samples, sampling_rate)

### Butterfilter from https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/12233959#12233959
def butter_bandpass(lowcut, highcut, fs, order=9):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def moving_average(a, N=3):
    #https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    cumsum = np.cumsum(np.insert(a, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_max_min(a, window_size, step_size):
    # https://stackoverflow.com/questions/51808573/running-window-of-max-min-in-a-numpy-array
    nrows = (a.size - window_size)//step_size + 1
    n = a.strides[0]
    s = as_strided(a, shape=(nrows, window_size), strides=(step_size*n, n))
    return s.ptp(1)

def calculate_rms(samples, sampling_rate):
    """Calculates the rms of a signal and returns the rms and times
    returns a tuple of 
        S: spectrogram magnitude
        rms: root mean square of input signl [[rms_values]]
        times: timestamps of rms"""
    S, phase = librosa.magphase(librosa.stft(samples))
    rms = librosa.feature.rms(S=S)
    times = librosa.times_like(rms)
    return (S, rms, times)

def smooth_rms(rms, times, n=30):
    """Smooths the rms values of a signal using a moving average, n indicates how many neighbors are considered
    for the average
    returns tuple (avg_rms, avg_times)
    avg_rms: average rms depending on considered neighbors np.ndarray [shape=(t,)]
    avg_tims: timestamps of average_rms values """
    averaged_rms = moving_average(rms[0], N=n)
    avg_times = moving_average(times, N=n)
    return (averaged_rms, avg_times)


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

def limit_operation_frequencies(samples, sampling_rate):
    """ Filters out frequencies > 256Hz and  > 1700Hz of provided signal"""
    filtered_samples = butter_bandpass_filter(samples, 256,1700, sampling_rate)
    return filtered_samples

def limit_beep_frequencies(samples, sampling_rate):
    """ Filters out frequencies > 2400 and  > 2500Hz of provided signal"""
    # test with 5000 for second beep test
    filtered_samples = butter_bandpass_filter(samples, 5100,5200, sampling_rate)
    # Frequency of normal beep
    #filtered_samples = butter_bandpass_filter(samples, 4800,4900, sampling_rate)
    return filtered_samples


def create_spectrogram(signal):
    # for wide add figsize=(7,3)
    fig, ax = plt.subplots()
    frequencies = librosa.stft(signal)
    img = librosa.display.specshow(librosa.amplitude_to_db(frequencies, ref=np.max),y_axis="linear", x_axis="time")
    # for startup circles
    #rect1 = patches.Rectangle((6.2, 256), 1.5, 330, linewidth=1, edgecolor='b', facecolor='none')
    #rect2 = patches.Rectangle((6.2, 730), 1.5, 330, linewidth=1, edgecolor='b', facecolor='none')
    #rect3 = patches.Rectangle((6.2, 1300), 1.5, 330, linewidth=1, edgecolor='b', facecolor='none')
    #ax.add_patch(rect1)
    #ax.add_patch(rect2)
    #ax.add_patch(rect3)
    #circle = patches.Ellipse((6,2450),0.8, 300,linewidth=1, edgecolor='b', facecolor='none')
    #ax.add_patch(circle)
    rect = patches.Rectangle((1.2, 5150), 1.5, 50, linewidth=1, edgecolor='b', facecolor='none')
    rect = patches.Rectangle((6, 4850), 0.5, 50, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel("Time [s]")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    #plt.show()
    plt.savefig("full_spectrogram.png",dpi=600)
    return None


def get_silent_action_start(sample, sampling_rate):
    """Extracts the beginning of an action in a sample with silent background"""
    # to adjust finding, play around with top_db
    parts = librosa.effects.split(sample, top_db=50, ref=np.max, frame_length=1024, hop_length=256)
    # start is in first part:
    start_frame = parts[0][0]
    start_time = parts[0][0]/sampling_rate
    return (start_frame, start_time)


def get_action_start(rms, rms_times, thresh=0.001, w=60, start=0):
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
    
def get_action_end(rms, rms_times, action_start_index, thresh=0.0004, w=50):
    """Returns index and end time of autmatically detected"""
    diffs= running_max_min(rms, w, 1)
    for i, j in enumerate(diffs):
        if i > action_start_index and j < thresh:
            return (i, rms_times[i+w-1])
    return (-1,-1)

def detect_events(rms, rms_times):
    events = [(0,0,0,0)]
    for i in range(20):        
        (start,start_time) = get_action_start(rms, rms_times,w=8,thresh=0.0007,start=events[-1][2])
        (ending, end_time) = get_action_end(rms, rms_times, start, w=8,thresh=0.0006)
        if start is -1:
            break
        events.append((start, start_time, ending, end_time))
    # for error beeps merge two events
    beeps = []
    for i in range(1, len(events)-1,2):
        beeps.append((events[i][0],events[i][1],events[i+1][2],events[i+1][3]))
    #return events[1:]
    print(events)
    print(beeps)
    return beeps


def plot_spectro_rms_smoothrms(S, rms, rms_times, avg_rms, avg_rms_times, f_name="figure"):
    """Creates RMS plot
        Input:
            S: spectrogram magnitude
            rms: rms for each frame
            rms_times: timestamps from librosa
            avg_rms: moving average rms from smoothness
            avg_rms_times: moving average timestamps from smoothness
        Output: Figure: rms_avg_<name>.png with three rows, rms clean, rms smooth, png"""
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    #(start,start_time) = get_action_start(avg_rms, avg_rms_times)
    #(ending, end_time) = get_action_end(avg_rms, avg_rms_times, start)
    # for beep
    #(start,start_time) = get_action_start(avg_rms, avg_rms_times,w=5,thresh=0.00013)
    # fan error
    (start,start_time) = get_action_start(avg_rms, avg_rms_times,w=8,thresh=0.0008)
    # beep
    #(ending, end_time) = get_action_end(avg_rms, avg_rms_times, start, w=17)
    #fan error
    (ending, end_time) = get_action_end(avg_rms, avg_rms_times, start, w=8,thresh=0.0006)
    events = detect_events(avg_rms, avg_rms_times)
    for i in events:
        if i[0] is -1:
            continue
        ax[0].axvline(x=i[1],c="grey",linestyle=":")
        ax[0].axvline(x=i[3], c="grey",linestyle="--")
        ax[1].axvline(x=i[1],c="grey",linestyle=":")
        ax[1].axvline(x=i[3],c="grey",linestyle="--")
        ax[2].axvline(x=i[1],c="grey",linestyle=":")
        ax[2].axvline(x=i[3],c="grey",linestyle="--")
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
    #ax[0].set(title='Log Power Spectrogram')
    # add start and end of action
    #ax[0].axvline(x=start_time, c="grey", linestyle=":")
    #ax[0].axvline(x=end_time, c="grey", linestyle="--")
    ax[1].semilogy(rms_times, rms[0], label='RMS Energy')
    # add start and end of action
    #ax[1].axvline(x=start_time, c="grey", linestyle=":")
    #ax[1].axvline(x=end_time, c="grey", linestyle="--")
    #ax[1].set(xticks=[])
    ax[1].legend(loc=1)
    ax[1].set_ylabel("W")
    #ax[1].set_ylim(ymin=10**(-4),ymax=2*10**(-3))# normal: ymin=2.5*10**(-3),ymax=7*10**(-3)
    # normal beep  at 4850Hz
    #ax[1].set_ylim(ymin=10**(-4), ymax=6*10**(-4))# normal: ymin=2.5*10**(-3),ymax=7*10**(-3)
    # fan error
    ax[1].set_ylim(ymin=3*10**(-4), ymax=5*10**(-3))
    ax[1].label_outer()
    ax[2].semilogy(avg_rms_times,avg_rms, label="Smoothed RMS Energy")
    # add start and end of action
    #ax[2].axvline(x=start_time, c="grey", linestyle=":")
    #ax[2].axvline(x=end_time, c="grey", linestyle="--")
    #ax[2].set(xticks=[])
    ax[2].legend(loc=1) # force top right
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("W")
    #ax[2].set_ylim(ymin=10**(-4),ymax=2*10**(-3)) # normal:  ymin=2.5*10**(-3),ymax=7*10**(-3)
    # normal beep  at 4850Hz
    #ax[2].set_ylim(ymin=10**(-4), ymax=6*10**(-4))
    # fan error
    ax[2].set_ylim(ymin=3*10**(-4), ymax=5*10**(-3))
    ax[0].label_outer()
    ax[2].xaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig("beep_full.png", dpi=600)
    #plt.show()

def plot_amplitude(samples, sampling_rate):
    # plot Loudness of signal
    plt.figure()
    librosa.display.waveshow(y=samples, sr = sampling_rate)
    #s = get_action_start(samples,sampling_rate)[1]
    #plt.axvline(x=s, c="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("waveshow.png")

def plot_spectrogram(S):
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time')#, ax=ax[0])
    #ax[0].set(title='Log Power Spectrogram')
    plt.title("Log-frequency power spectrogram")
    plt.xlabel("Time (s)")
    #plt.show()
    plt.savefig("log_spectrogram.png")

def plot_spectrogram_with_actions(S, start, ending):
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
    ax.set(title='Log Power Spectrogram')
    ax.axvline(x=start, c="black",linestyle=".")
    ax.axvline(x=ending, c="black")
    ax.set_xlabel("Time (s)")
    plt.savefig("log_spectrogram_start_ending.png")

def plot_flatness(sample, sr, S):
    flatness = librosa.feature.spectral_flatness(y=sample, S=S)
    print(flatness)
    plt.figure()
    plt.plot(flatness[0])
    plt.show()
    pass

def plot_rms(rms, rms_times):
    plt.figure()
    plt.semilogy(rms_times, rms, label='RMS Energy')
    plt.title("Smoothed RMS Energy")
    #plt.axvline(6.072018140589573, c="green")
    #plt.axvline(7.9063945578231145, c="red")
    # add start and end of action
    plt.savefig("rms.png", dpi=600)

def plot_spectrogram_delta(signal, sampling_rate):
    fig, ax = plt.subplots(2,1)
    frequencies = librosa.stft(signal)
    delta = librosa.feature.delta(frequencies,order=4)
    librosa.display.specshow(librosa.amplitude_to_db(frequencies, ref=np.max),y_axis="linear", x_axis="time", ax=ax[0])
    librosa.display.specshow(librosa.amplitude_to_db(delta, ref=np.max),y_axis="linear", x_axis="time", ax=ax[1])
    plt.show()

def main(args):
    (samples, sampling_rate)= read_file(f_path+"server_error_cut.wav")
    duration = len(samples)/sampling_rate    
    # filter signal for beeps
    filtered = limit_beep_frequencies(samples, sampling_rate)
    print(filtered[0])
    a = librosa.amplitude_to_db(filtered, ref=np.max)
    print(a[0])
    # filter normal 
    #filtered = limit_operation_frequencies(samples,sampling_rate)
    (S, rms, rms_times) = calculate_rms(filtered, sampling_rate)
    #for beep
    #(avg_rms, avg_rms_times) = smooth_rms(rms, rms_times,n=10)
    # for fan error
    (avg_rms, avg_rms_times) = smooth_rms(rms, rms_times,n=9)
    #for normal
    #(avg_rms, avg_rms_times) = smooth_rms(rms, rms_times)
    #(start,start_time) = get_action_start(avg_rms, avg_rms_times)
    #(ending, end_time) = get_action_end(avg_rms, avg_rms_times, start)
    #create_spectrogram(samples)
    #plot_spectro_rms_smoothrms(S,rms,rms_times,avg_rms,avg_rms_times)
    #plot_amplitude(filtered,sampling_rate)
    #plot_flatness(filtered, sampling_rate ,S)
    #plot_spectrogram_with_actions(S, start_time, end_time)
    #plot_rms(rms[0], rms_times)
    #create_amplitude_plot(filtered, sampling_rate)
    #plot_spectrogram(filtered)
    # TODO evaluate if log mel spectrograms provide insights
    plot_spectrogram_delta(samples, sampling_rate)

if __name__ == "__main__":
    main(sys.argv)