#!/usr/bin/env python3.8
# coding: utf-8

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.append("../../") # necessary to import sidechannel
from sidechannel.audio.reference_building import get_rms_change
from sidechannel.audio.sound_module.plotting import plot_rms_comparison

def compare_signal_length(rms_diffs1, rms_diffs2):
    if len(rms_diffs1) != len(rms_diffs2):
        print("RMS diffs are of different length, can't compare")
        print("Len difflist 1: {}".format(len(rms_diffs1)))
        print("Len difflist 2 : {}".format(len(rms_diffs2)))
        return False
    return True

def calculate_signal_difference(rms_diffs1, rms_diffs2):
    """
        calculates the general signal of two rms traces and returns the error
        in each frame between the two signals.
    """
    #rms_diffs1 = get_rms_change(file1)[0]
    #rms_diffs2 = get_rms_change(file2)[0]
    if not compare_signal_length(rms_diffs1, rms_diffs2):
        return None
    signal1 = []
    signal2 = []
    s1 = 0
    s2 = 0
    error = []
    for i in range(len(rms_diffs1)):
        s1 += rms_diffs1[i]
        s2 += rms_diffs2[i]
        signal1.append(s1)
        signal2.append(s2)
        error.append((signal1[i]-signal2[i]) ** 2)
    #print("max error signal comp:", max(error))
    #t = 15/len(signal1)
    #times = [i*t for i in range(len(signal1))]
    #print(signal1)
    #print(signal2)
    #plot_rms_comparison(signal1, signal2, error, times, viz=True)
    return (signal1, signal2, error)

def compare_rms_changes(rms_diffs1, rms_diffs2):
    """Compares the change in signal of two files"""
    #rms_diffs1 = get_rms_change(file1)[0]
    #rms_diffs2 = get_rms_change(file2)[0]
    if not compare_signal_length(rms_diffs1, rms_diffs2):
        return None
    error = []
    for i in range(len(rms_diffs1)):
        error.append((rms_diffs1[i]-rms_diffs2[i]) ** 2)
    #print("max error signal changes:", max(error))
    return error

def naive_similarity_score(error, thresh=0.0):
    n = len(error)
    acc = 0
    for e in error:
        if e <= thresh:
            acc+= 1
    return acc/n


def validate_rms_signals(file1, file2):
    diffs = calculate_signal_difference(file1, file2) # (diffs1, diffs2, error)
    #th = 5.438883497907486e-05 (max from a reference)
    th = 3.0e-07
    similarity_score = naive_similarity_score(diffs[2], th)
    return similarity_score


def validate_rms_diffs(diffs1, diffs2):
    th = 2.0e-09 # chosen arbitaritily by analyzing different diff changes with a single file
    error = compare_rms_changes(diffs1, diffs2)
    similarity_score = naive_similarity_score(error, th)
    return similarity_score


def main(args):
    test = calculate_signal_difference("../../reference_1649409699212828715.wav", "../../reference_1649409699212828715.wav")
    print(naive_similarity_score(test[2]))
    pass

if __name__=="__main__":
    main(sys.argv)