#!/usr/bin/env python3.7
# coding: utf-8

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from librosa import display



def plot_spectrogram(signal, patches=[], f_name="signal", scale="log", viz=False):
    # for wide add figsize=(7,3)
    fig, ax = plt.subplots()
    frequencies = librosa.stft(signal, n_fft=4096)
    img = librosa.display.specshow(librosa.amplitude_to_db(frequencies, ref=np.max),y_axis=scale, x_axis="time", sr=44100)#, sr=44100)
    # add markers if present
    if len(patches) > 0:
        for p in patches:
            ax.add_patch(p)
    ax.set_xlabel("Time [s]")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    if viz:
        plt.show()
    else: 
        plt.savefig("{}_{}_spectrogram.pdf".format(f_name,scale))
    return None

def plot_spectrogram_rms_smoothrms(S, rms, rms_times, avg_rms, avg_rms_times, events=[], y_min=None,y_max=None, f_name="figure", viz=False):
    """Creates RMS plot
        Input:
            S: spectrogram magnitude
            rms: rms for each frame
            rms_times: timestamps from librosa
            avg_rms: moving average rms from smoothness
            avg_rms_times: moving average timestamps from smoothness
            y_min: for bottom two plots: shared y min of y axis
            y_max: for bottom two plots: shared y max of y axis
        Output: Figure: rms_avg_<name>.png with three rows, rms clean, rms smooth, png"""
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    if len(events)>0:
        for i in events:
            if i[0] == -1:
                continue
            ax[0].axvline(x=i[1],c="grey",linestyle=":")
            ax[0].axvline(x=i[3], c="grey",linestyle="--")
            ax[1].axvline(x=i[1],c="grey",linestyle=":")
            ax[1].axvline(x=i[3],c="grey",linestyle="--")
            ax[2].axvline(x=i[1],c="grey",linestyle=":")
            ax[2].axvline(x=i[3],c="grey",linestyle="--")
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
    ax[1].semilogy(rms_times, rms[0], label='RMS Energy')
    ax[1].legend(loc=1)
    ax[1].set_ylabel("W")
    if y_min is not None and y_max is not None:
        ax[1].set_ylim(ymin=y_min, ymax=y_max)
        ax[2].set_ylim(ymin=y_min, ymax=y_max)    
    ax[1].label_outer()
    ax[2].semilogy(avg_rms_times,avg_rms, label="Smoothed RMS Energy")
    ax[2].legend(loc=1) # force top right
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("W")
    ax[0].label_outer()
    ax[2].xaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    if viz:
        plt.show()
    else:
        plt.savefig("{}_triple.png".format(f_name), dpi=1200)
    return None

def plot_double_spectrogram(y1, y2, f_name="figure", viz=False):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].minorticks_on()
    ax[1].minorticks_on()
    frequencies1 = librosa.stft(y1)
    img1 = librosa.display.specshow(librosa.amplitude_to_db(frequencies1, ref=np.max), y_axis="log", x_axis="time", ax=ax[0])
    ax[0].set(title='Observed Signal')
    frequencies2 = librosa.stft(y2)
    img2 = librosa.display.specshow(librosa.amplitude_to_db(frequencies2, ref=np.max), y_axis="log", x_axis="time", ax=ax[1])
    ax[1].set(title='Constructed Reference Signal')
    for ax_i in ax:
        ax_i.label_outer()
    fig.colorbar(img1, ax=[ax[0], ax[1]])
    ax[1].set_xlabel("Time [s]")
    #plt.tight_layout()
    if viz:
        plt.show()
    else:
        plt.savefig("{}_double.pdf".format(f_name), bbox_inches="tight")
    return None

def plot_rms_diff(rms, rms_diff, rms_times):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    #ax[0].semilogy(rms_times, rms, label="Smoothed RMS Energy")
    ax[0].plot(rms_times, rms, label="Smoothed RMS Energy")
    ax[0].legend(loc=1)
    ax[1].plot(rms_times,rms_diff, label="RMS Action diff")
    ax[1].legend(loc=1)
    for ax_i in ax:
        ax_i.label_outer()
    plt.show()


def plot_pure_spectrogram(signal,scale="log", f_name="pure_spec"):
    # https://stackoverflow.com/questions/38411226/matplotlib-get-clean-plot-remove-all-decorations
    frequencies=librosa.stft(signal)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    #img = librosa.display.specshow(librosa.amplitude_to_db(frequencies),ref=np.max, y_axis="log")
    img = librosa.display.specshow(librosa.amplitude_to_db(frequencies, ref=np.max), y_axis=scale, x_axis="time", ax=ax)
    # remove white space around the spectrogram
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig("{}.png".format(f_name), bbox_inches=extent, dpi=900)

def plot_rms_comparison(constructed_signal, real_signal, error, times, f_name="signal_comparison.pdf", viz=False):
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax[0].plot(times, constructed_signal, "b-", label="Constructed Signal")
    ax[0].plot(times, real_signal, "r--", label="Recorded Signal")
    ax[0].legend()
    # color in area between real and contructed signal
    #ax[0].fill_between(times, real_signal, constructed_signal)
    #error bar
    ax[1].plot(times, error, "r-", label="Error")
    ax[1].legend()
    acc_e = []
    e = 0
    for i in range(len(error)):
    # TODO: accumulative error
        e += error[i]
        acc_e.append(e)
    ax[2].plot(times, acc_e, label="Accumulated Error")
    ax[2].legend()
    if viz:
        plt.show()
    plt.savefig(f_name, bbox_inches="tight")
    pass

def plot_opencv_image(image_data, viz=True):
    plt.imshow(image_data)
    plt.show()

def plot_dwt(Y, D, wp):
    # from https://librosa.org/doc-playground/0.9.1/generated/librosa.sequence.dtw.html?highlight=dtw#id1
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames', ax=ax[0])
    ax[0].set(title='DTW cost', xlabel='F2', ylabel='F1')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2], title='Matching cost function')
    plt.show()
    pass

