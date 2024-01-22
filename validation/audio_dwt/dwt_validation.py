#!/usr/bin/env python3.8
# coding: utf-8

import sys
import os
import inspect
import librosa
from fastdtw import fastdtw

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.append("../../") # necessary to import sidechannel
from sidechannel.audio.sound_recording import read_file
from sidechannel.audio.sound_module.plotting import plot_dwt

def calculate_dtw(y1, sr1, y2, sr2):
    """y1: loaded signal1
    sr1: sample rate of the first signal
    y2, loaded signal2
    sr2: sample rate of the second signal """
    Y1 = librosa.feature.chroma_cens(y=y1, sr=sr1)
    Y2 = librosa.feature.chroma_cens(y=y2, sr=sr2)
    D, wp = librosa.sequence.dtw(Y1, Y2, subseq=True)
    matching_cost = D[-1, :] / wp.shape[0]
    #print(matching_cost)
    #plot_dwt(Y1, D, wp)
    return D, wp, matching_cost

def naive_similarity_score(dist):
    # naive first implementation: take min from the minimum distance, if distance is >1, return 0
    #print(cost_array)
    scale = 0.001 # needs to be determined experimentally
    return max(1.0-dist*scale,0)

def compare_dtw(file1, file2):
    (y1, sr1) = read_file(file1)
    (y2, sr2) = read_file(file2)
    D, wp, matching_cost = calculate_dtw(y1, sr1, y2, sr2)
    similarity_score = naive_similarity_score(matching_cost)
    return similarity_score
   
def compare_fast_dtw(file1, file2):
    (y1, sr1) = read_file(file1)
    (y2, sr2) = read_file(file2)
    dist, path = fastdtw(y1, y2)
    similarity_score = naive_similarity_score(dist)
    return similarity_score

def main(args):
    pass


if __name__ == "__main__":
    main(sys.argv)