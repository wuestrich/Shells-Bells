#!/usr/bin/env python3.8
# coding: utf-8


import sys
import subprocess
import json
import pathlib
from reference_creation import get_rms_change
from validation.audio_rms.rms_validation import validate_rms_diffs, validate_rms_signals
from validation.audio_dwt.dwt_validation import compare_fast_dtw
from validation.ml.cnn_validation import cnn_validation
from sidechannel.audio.sound_module.change_extraction import highlight_significant_changes


def validate_rms(diffs1, diffs2, threshold, verbose=False):
    """returns a similarity score based on the error between two signals"""
    score = validate_rms_signals(diffs1, diffs2)
    #score = 1.0 - min(sum(validation[2])*2,1)
    print("Similarity score: {}".format(score))
    if score >= threshold:
        return (True, score)
    else: 
        return (False, score)


def validate_rms_changes(diffs1, diffs2, threshold, verbose=False):
    score = validate_rms_diffs(diffs1, diffs2)
    #score = 1.0 - min(sum(validation[2])*2,1)
    print("Similarity score: {}".format(score))
    if score >= threshold:
        return (True, score)
    else: 
        return (False, score)


def validate_dtw(reference_file, comparison_file, threshold):
    score = compare_fast_dtw(reference_file, comparison_file)
    print("Similarity score: {}".format(score))
    if score >= threshold:
        return (True, score)
    else:
        return (False, score)


def compare_audfprint(fp_file, new_file, illustrate=False):
    f_loc = pathlib.Path(__file__)
    python_env_path = "{}/bin/python3.8".format(f_loc.parent.resolve())
    audfprint = "{}/audfprint/audfprint.py".format(f_loc.parent.resolve())
    command = [python_env_path, audfprint,fp_file] # pay attention to paths!!!!


def validate_cnn(recording_file:str, constructed_file:str, n_exepected_events:int=0):
    # apply significant change extraction to both files
    print(f"Validating files via CNN: {recording_file}, {constructed_file}")
    result = cnn_validation(recording_file, constructed_file)
    # classify input
    return result

########## Other Utils 
def verify_rms(rms_diffs1, rms_diffs2):
    """calculates and compares two signals caused by rms_diffs"""
    if len(rms_diffs1) != len(rms_diffs2):
        print("RMS diffs are of different length, can't compare")
        print("Len difflist 1: {}".format(len(rms_diffs1)))
        print("Len difflist 2 : {}".format(len(rms_diffs2)))
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
        error.append(abs(signal1[i]-signal2[i]))
    return (signal1, signal2, error)


def main(args):
    pass


if __name__== "__main__":
    main(sys.argv)