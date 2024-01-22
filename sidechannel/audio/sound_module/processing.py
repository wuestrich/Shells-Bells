#!/usr/bin/env python3.7
# coding: utf-8

import sys
import signal_analysis
import plotting
import matplotlib.patches as patches
import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from sound_recording import read_file

def noms():
    """Plot all NOMS paper figures and configurations
    Plots generated: 
        - Spectrogram of the fan_on with fans and beep circled
        - triple fig of beep in fan_on
        - triple fig of fan_error_detection 
    """
    # startup processing
    (signal, sampling_rate) = read_file("../data/sound/power_on3.wav")
    np.savetxt("audio_time_series.csv",signal, delimiter=",")
    print(sampling_rate)
    # rectangles and circles for startup marks
    rect1 = patches.Rectangle((6.2, 256), 1.5, 330, linewidth=1, edgecolor='b', facecolor='none')
    rect2 = patches.Rectangle((6.2, 730), 1.5, 330, linewidth=1, edgecolor='b', facecolor='none')
    rect3 = patches.Rectangle((6.2, 1300), 1.5, 330, linewidth=1, edgecolor='b', facecolor='none')
    circle = patches.Ellipse((6,2450),0.8, 300,linewidth=1, edgecolor='b', facecolor='none')
    plotting.plot_spectrogram(signal, [rect1,rect2,rect3,circle],"power_on3","log",viz=False)
    # fan accelleration
    signal_fans = signal_analysis.limit_frequencies(signal, sampling_rate, 256,1700)
    plotting.plot_spectrogram(signal_fans, [], "power_on_limited_frequencies", "log", viz=False)
    (S, rms, rms_times) = signal_analysis.rms_energy(signal_fans, sampling_rate)
    np.savetxt("fan_on_rms.csv", rms, delimiter=",")
    np.savetxt("fan_on_rms_times.csv", rms_times, delimiter=",")
    (avg_rms, avg_rms_times) = signal_analysis.rms_energy_smoothed(rms, rms_times, n=30)
    np.savetxt("fan_on_avg_rms.csv", avg_rms, delimiter=",")
    np.savetxt("fan_on_avg_rms_times.csv", avg_rms_times, delimiter=",")
    (start,start_time) = signal_analysis.get_event_start(avg_rms, avg_rms_times, thresh=0.001, w=60)
    (ending, end_time) = signal_analysis.get_event_end(avg_rms, avg_rms_times, start, thresh=0.0004, w=50)
    event = (start, start_time,ending,end_time)
    y_min_fans = 2.5*10**(-3)
    y_max_fans = 7*10**(-3) #y limits for the RMS energy plots, same for the lower parts for comparability
    plotting.plot_spectrogram_rms_smoothrms(S, rms, rms_times, avg_rms, avg_rms_times, [event], y_min_fans, y_max_fans, f_name="power_on3_fans", viz=False)
    ### Startup beep/buzzer
    signal_beep = signal_analysis.limit_frequencies(signal, sampling_rate,4800,4900)
    (S, rms, rms_times) = signal_analysis.rms_energy(signal_beep, sampling_rate)
    (avg_rms, avg_rms_times) = signal_analysis.rms_energy_smoothed(rms, rms_times, n=10)
    (start,start_time) = signal_analysis.get_event_start(avg_rms, avg_rms_times,w=5,thresh=0.00013)
    (ending, end_time) = signal_analysis.get_event_end(avg_rms, avg_rms_times, start, w=17)
    event = (start, start_time,ending,end_time)
    y_min_beep = 10**(-4)
    y_max_beep = 6*10**(-4)
    plotting.plot_spectrogram_rms_smoothrms(S, rms, rms_times, avg_rms, avg_rms_times, [event], y_min_beep, y_max_beep, f_name="power_on3_buzzer", viz=False)
    # fan_error detection
    (signal, sampling_rate) = read_file("../data/sound/server_error_cut.wav")
    signal_fan_error = signal_analysis.limit_frequencies(signal,sampling_rate,5100,5200)
    (S, rms, rms_times) = signal_analysis.rms_energy(signal_fan_error, sampling_rate)
    (avg_rms, avg_rms_times) = signal_analysis.rms_energy_smoothed(rms, rms_times,n=9)
    (start,start_time) = signal_analysis.get_event_start(avg_rms, avg_rms_times,w=8,thresh=0.0008)
    (ending, end_time) = signal_analysis.get_event_end(avg_rms, avg_rms_times, start, w=8,thresh=0.0006)
    events = signal_analysis.detect_events(avg_rms, avg_rms_times)
    y_min_fan_error = 3*10**(-4)
    y_max_fan_error = 5*10**(-3)
    plotting.plot_spectrogram_rms_smoothrms(S, rms, rms_times, avg_rms, avg_rms_times, events, y_min_fan_error, y_max_fan_error, f_name="fan_error", viz=False)


def main(args):
    noms()


if __name__ == "__main__":
    main(sys.argv)