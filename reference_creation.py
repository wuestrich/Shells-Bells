#! /usr/bin/env python3.7
# coding: utf-8

import subprocess
import ast
import random
import sys
import sox
import sidechannel.audio.sound_recording
import pandas as pd
import time
import os
import sidechannel.audio.sound_module.signal_analysis as signal_analysis
from sidechannel.audio.sound_module.plotting import plot_spectrogram, plot_spectrogram_rms_smoothrms, plot_rms_diff
import csv
from sidechannel.audio.sound_recording import store_audio, read_file
#from testing.test_verification import generate_random_reference

referenceDB="asdf/modeldb.csv"

def generate_command_reference(sound, event, sr, end_ts, reference_path=""):
    """Generates a reference from a detected event and an accompanying sound"""
    # TODO Refactor audio-name, this is way too complex at the moment
    reference = {}
    reference["device"] = event["device"]
    reference["netFN"] = event["netFN"]
    reference["command_bytes"] = event["command_bytes"]
    f_name = f"{reference['device']}_{reference['netFN']}_{reference['command_bytes']}" 
    # 1. store sound recording to file
    audio_f_name = "{}/referenceSound/{}".format(reference_path,f_name)# relevant suffixes that have the sound _action_{}.wav, _start_noise_profile, .wav, 
    # filtering for fans
    filtered_audio = signal_analysis.limit_frequencies(sound, sr, 400, 900)
    in_audio = store_audio(filtered_audio, sr, audio_f_name)
    #plot_spectrogram(read_file(in_audio)[1],f_name="reference_building")
    # 2. determine length of action
    meta = get_action_length(in_audio, False) # meta[0]=duration, meta[1]= start_ts (relative from beginning of recording) meta[2]=end_ts (relative from beginning of recording)
    if meta[1] == -1 or meta[2] == -1:
        print(meta)
        print("ERROR: Failed to determine length of action")
        return None
    # 3. generate cleaned reference of action
    reference["duration"]=meta[0]
    beginnig = extract_start(meta[1], in_audio)
    noise_prof = build_noise_profile(beginnig)
    tfm = sox.Transformer()
    tfm.noisered(profile_path=noise_prof, amount = 0.12)
    tfm.trim(meta[1],meta[2]) # split remove sound before and after action
    cleaned_file = "{}_action_{}.wav".format(audio_f_name.split(".wav")[0],0.05)
    tfm.build(input_filepath="{}.wav".format(audio_f_name),output_filepath=cleaned_file)
    # 4. calculate from command to time to action (tta)
    # tta = action_start_ts - command_ts
    # action_start_ts = end_ts - len(recording) + start_action
    action_start_ts = end_ts - ((len(sound)/sr)*1000000000) + meta[1]*1000000000
    reference["tta"] = (action_start_ts - event["ts"]) / 1000000000
    reference["sound_file_prefix"] = audio_f_name
    # store to csv
    with open("{}/modeldb.csv".format(reference_path), "a") as f:
        w = csv.DictWriter(f, reference.keys())
        #w.writeheader()
        w.writerow(reference)
    #print(reference)



    #TODO include response code in reference
    return 1

def generate_rms_command_reference(sound, event, sr, end_ts, reference_path=""):
    reference = {}
    reference["device"] = event["device"]
    reference["netFN"] = event["netFN"]
    reference["command_bytes"] = event["command_bytes"]
    f_name = "{}_{}_{}".format(reference["device"],reference["netFN"],reference["command_bytes"]) 
    audio_f_name = "{}/referenceSound/{}".format(reference_path,f_name)# relevant suffixes that have the soud _action_{}.wav, _start_noise_profile, .wav, 
    filtered_audio = signal_analysis.limit_frequencies(sound, sr, 400, 900)
    in_audio = store_audio(filtered_audio, sr, audio_f_name)
    meta = get_action_length(in_audio, False) # meta[0]=duration, meta[1]= start_ts (relative from beginning of recording) meta[2]=end_ts (relative from beginning of recording)
    if meta[1] == -1 or meta[2] == -1:
        print(meta)
        print("ERROR: Failed to determine length of action")
        return None
    reference["duration"]=meta[0]
    (diffs, diff_times) = get_rms_change(in_audio, meta[1],meta[2])
    # 4. generate cleaned reference of action
    beginnig = extract_start(meta[1], in_audio)
    noise_prof = build_noise_profile(beginnig)
    tfm = sox.Transformer()
    tfm.noisered(profile_path=noise_prof, amount = 0.12)
    tfm.trim(meta[1],meta[2]) # split remove sound before and after action
    cleaned_file = "{}_action_{}.wav".format(audio_f_name.split(".wav")[0],0.05)
    tfm.build(input_filepath="{}.wav".format(audio_f_name),output_filepath=cleaned_file)
    # 4. calculate from command to time to action (tta)
    #tta = action_start_ts - command_ts
    #action_start_ts = end_ts - len(recording) + start_action
    action_start_ts = end_ts - ((len(sound)/sr)*1000000000) + meta[1]*1000000000
    reference["tta"] = (action_start_ts - event["ts"]) / 1000000000
    reference["sound_file_prefix"] = audio_f_name
    reference["rms_diffs"] = diffs
    reference["rms_diff_times"] = diff_times
    # store to csv
    with open("{}/modeldb.csv".format(reference_path), "a") as f:
        w = csv.DictWriter(f, reference.keys())
        #w.writeheader()
        w.writerow(reference)
    #print(reference)

def get_action_length(f_name, viz=False):
    """Uses NOMS algorithm to determine the length of an action
    input:
        f_name: filename of recording which contains the length (.wav)
    output:
        (length, start, end): tuple that contains 
            - length: length of action (end-start time)
            - start timestamp of action in the file
            - end timestamp of action in the file """
    (y, sr) = read_file(f_name)
    (S, rms, times) = signal_analysis.rms_energy(y, sr)
    # windows and threshhold selected for fan spinup
    (avg_rms, avg_rms_times) = signal_analysis.rms_energy_smoothed(rms, times, n=60)
    (start,start_time) = signal_analysis.get_event_start(avg_rms, avg_rms_times, thresh=0.0007, w=60)
    (end, end_time) = signal_analysis.get_event_end(avg_rms, avg_rms_times, start, thresh=0.0004, w=50)
    duration = end_time-start_time
    if viz:
        plot_spectrogram_rms_smoothrms(S, rms, times, avg_rms, avg_rms_times, f_name="reference_building")
    return (duration,start_time, end_time)



def get_rms_change(f_name, start=None, end=None, viz=False):
    """Returns the RMS change due to activity in the input file
    input:
        f_name: filename from which the diff should be extracted
        start: optional: if set specify start timestamp for analysis
        end: optional if set specifies end timestamp for analysis
    output
        (rms_diff, rms_diff_times): differences to base signal, and timestamps at which these happen"""
    (y,sr) = read_file(f_name)
    (S, rms, times) = signal_analysis.rms_energy(y,sr)
    (avg_rms, avg_rms_times) = signal_analysis.rms_energy_smoothed(rms, times, n=60)
    if not start:
        start = 0
    if not end:
        end = avg_rms_times[-1]
    start_index = next(x[0] for x in enumerate(avg_rms_times) if x[1] > start)
    end_index = next(x[0] for x in enumerate(avg_rms_times) if x[1] >= end)
    rms_diff = []
    rms_diff_times = []
    for j in range(start_index, end_index):
            rms_diff.append(avg_rms[j]-avg_rms[j-1])
            rms_diff_times.append(avg_rms_times[j]-avg_rms_times[start_index])
    return (rms_diff, rms_diff_times)

def sample_denoise(f_name, start, end):
    #### WIP #########
    # Step 1: get beginning of action (e.g. by using the RMS method)
    (y, sr) = read_file(f_name)
    #### Pre-bandpass filter with librosa
    #y = signal_analysis.limit_frequencies(y, sr, 256,1700)
    #sf.write("{}_filtered.wav".format(f_name.split(".wav")[0]), y, sr, subtype="PCM_24")
    #(y, sr) = signal_analysis.read_file("{}_filtered.wav".format(f_name.split(".wav")[0]))
    #### End Pre-bandpass filter
    # Step 2: write stuff happening before an actual action a file
    start_file = extract_start(start,f_name)
    # Step 3: calculate noise profile from file
    noise_prof = build_noise_profile(start_file)
    # Step 4: denoise initial file with noise profile
    tfm = sox.Transformer()
    # loop to determine best values for sox denoise
    tfm.noisered(profile_path=noise_prof, amount = 0.12)
    cleaned_file = "{}_denoised_{}.wav".format(f_name.split(".wav")[0],0.12)
    tfm.build(input_filepath=f_name,output_filepath=cleaned_file)
    
def extract_start(time, f_name):
    """Writes the start of a file to a new file up to a specified timestamp
    Input: 
        f_name: filename of recording from which the beginning should be extracted
        time: timestamp until which the recoring should take place
    Output: 
        out_f_name: filename of output file"""
    tfm = sox.Transformer()
    tfm.trim(0.0,time) # time -1 just to mitigate potential slow detection beginning
    out_file = "{}_start.wav".format(f_name.split(".wav")[0])
    tfm.build(input_filepath=f_name, output_filepath=out_file)
    return out_file

def build_noise_profile(f_name):
    tfm = sox.Transformer()
    profile_fpath = "{}_noiseprofile".format(f_name.split(".wav")[0])
    tfm.noiseprof(input_filepath=f_name, profile_path=profile_fpath)
    return profile_fpath


################################# 
# DYNAMIC REFERENCE GENERATION  #
#################################

def sox_generate_clean_file(period, f_name="silence"):
    command = ["sox", "-n", "-r", "44100" ,"{}.wav".format(f_name), "trim", "0.0", "{}".format(period)]
    ret = subprocess.run(command, capture_output=False)
    return "{}.wav".format(f_name)


def generate_clean_file(period, f_name="silence"):
    """Generates a silent sound file that has length of period
    Suggestion: len(buffer)+max(tta+duration) for any action in the reference set
    Input:
        period: time in 10^-2 seconds"""
    #print(period)
    out = sound_recording.store_audio([0.0]*441*period,44100, f_name)
    return out

def sox_prepend_silence(period, sound):
    """Prepends silent part before a sound
    Input: 
        period: length of silence before the sound, sample rate of 44100
            unity of period is s, e.g. 10.2
        sound: path to sound file name to which the silent part should be prepended"""
    silent_prefix = sox_generate_clean_file(period, "prepend")
    tfm = sox.Combiner()
    ref_name = "silent_prefix_ref.wav"
    tfm.build(input_filepath_list=[silent_prefix,sound],output_filepath=ref_name,combine_type="concatenate")
    os.remove(silent_prefix)
    return ref_name

def prepend_silence(period, sound):
    """Prepends silent part before a sound
    Input: 
        period: length of silence before the sound, sample rate of 44100
            unity of period is 10^-2s = cs
        sound: path to sound file name to which the silent part should be prepended"""
    silent_prefix = generate_clean_file(period)
    tfm = sox.Combiner()
    ref_name = "silent_prefix_ref.wav"
    tfm.build(input_filepath_list=[silent_prefix,sound],output_filepath=ref_name,combine_type="concatenate")
    return ref_name

def sox_add_reference(base_file, reference_to_add, ts):
    """ add a sound to an existing sound file at a given timestamp (relative from beginning of a file)"""
    ref_at_ts = sox_prepend_silence(ts, reference_to_add)
    tfm = sox.Combiner()
    out = "ref_{}.wav".format(ts)
    tfm.build(input_filepath_list=[base_file, ref_at_ts], output_filepath=out, combine_type="mix")
    return out

def add_reference(base_file, reference_to_add, ts):
    """ add a sound to an existing sound file at a given timestamp (relative from beginning of a file)"""
    ref_at_ts = prepend_silence(ts, reference_to_add)
    tfm = sox.Combiner()
    out = "ref_{}.wav".format(ts)
    tfm.build(input_filepath_list=[base_file, ref_at_ts], output_filepath=out, combine_type="mix")
    return out

def trim_reference(reference_file, start, end):
    tfm = sox.Transformer()
    out = "trimmed_{}.wav".format(reference_file)
    tfm.trim(start, end)
    tfm.build(input_filepath=reference_file,output_filepath=out)
    os.replace(out, reference_file)
    return reference_file

def load_model_db(model_db_csv):
    """loads a model-db from a csv"""
    model_db = pd.read_csv(model_db_csv)
    return model_db


def check_reference(event, model_db):
    """Checks if a given event is in a given model-db and return index of matching row"""
    return model_db.index[(model_db["device"] == event["device"]) & (model_db["netFN"]==event["netFN"]) & (model_db["command_bytes"]==event["command_bytes"])].tolist()


def generate_dynamic_reference(events, start_time, length, model_db, model_db_dir="./", offset=0):
    """Generates a complete reference for a time-window
    Input:
        events: list of events that should be included in the reference
        start_time: start-time of the reference
        length: length of the reference in s
        offset: if the reference needs to be trimmed. time in 10^-2s until the actual audio in the real world starts
    Output:
        f_name: f_name of file which stores the complete reference"""
    total_length = length *1.0 + offset 
    reference = sox_generate_clean_file(total_length, "reference_{}".format(start_time))
    for event in events:
        #print("Processing event: ", event)
        ref_check = check_reference(event,model_db)
        if len(ref_check)==0: # 1. Check if event is in reference_db,  if not goto next event, if exists:
            print("Event has no reference, continuing")
            continue
        relative_ts = event["ts"] - start_time + int(model_db.loc[ref_check[0]]["tta"]*10**9) # 2. get ts of command by command_ts - start_time  + tta (convert s to ns)  
        # relative ts is in ns, convert+round to 10^-2s due to limited accuracy of sample rate
        rounded_relative_ts = relative_ts * 10 ** -9
        #print("RELATIVE TS OF EVENT: {}".format(rounded_relative_ts))
        ref_file = "{}{}_action_0.05.wav".format(model_db_dir, model_db.loc[ref_check[0]]["sound_file_prefix"])# get ref
        added = sox_add_reference(reference, ref_file, rounded_relative_ts) # 3.2 add reference with prepended silence
        os.replace(added, reference)
    # trim to length of audio-buffer to compare
    reference = trim_reference(reference, offset*1.0, total_length*1.0)
    return reference


def rms_resample(diff_list, samples):
    """Resamples a list to a new length based on the amount of samples and interpolates if necessary"""
    size = len(diff_list)
    if samples == size:
        return diff_list
    elif samples > size:
        # we randomly add zeroes in between
        i = random.randint(0,size-1)
        new_list = diff_list[:i] + [0] + diff_list[i:]
        return rms_resample(new_list, samples)
    else:
        # we need to accumulate the diffs into the number of samples
        i = random.randint(0, size-2)
        new_list = diff_list[:i] + [diff_list[i] + diff_list[i+1]] + diff_list[i+2:]
        return rms_resample(new_list, samples)

def generate_dynamic_rms_reference(events, start_time, length, model_db, reference_polls, offset=0):
    """Generates a complete rms reference for a time-window
    Input:
        events: list of events that should be included in the reference
        start_time: start-time of the reference
        length: length of the reference in s
        reference_polls: number of polls off the reference 
        offset: if the reference needs to be trimmed. time in 10^-2s until the actual audio in the real world starts
    Output:
        rms_reference: array that indicates the expected rms diffs"""
    s = sox_generate_clean_file(length*1.0 + offset)
    (reference_diffs, reference_diff_times) = get_rms_change(s)
    total_polls = len(reference_diffs)
    for event in events:
        #print("Processing event: ", event)
        ref_check = check_reference(event,model_db)
        if len(ref_check)==0: # 1. Check if event is in reference_db,  if not goto next event, if exists:
            print("Event has no reference, continuing")
            continue
        relative_ts = event["ts"] - start_time + int(model_db.loc[ref_check[0]]["tta"]*10**9) # 2. get ts of command by command_ts - start_time  + tta (convert s to ns)  
        relative_ts_second = relative_ts * 10 ** -9
        event_duration = model_db.loc[ref_check[0]]["duration"]
        # relative ts is in ns, convert+round to 10^-2s = cs due to limited accuracy of sample rate
        #rounded_relative_ts = round(relative_ts * 10 ** -7)
        event_diffs = ast.literal_eval(model_db.loc[ref_check[0]]["rms_diffs"])
        event_times = ast.literal_eval(model_db.loc[ref_check[0]]["rms_diff_times"])
        # deterimene startindex
        try:
            start_index = next(x[0] for x in enumerate(reference_diff_times) if x[1] >= relative_ts_second)
        except:
            start_index=len(reference_diff_times)-1
        samples = next(x[0] for x in enumerate(reference_diff_times) if x[1] >= event_duration)
        diffs = rms_resample(event_diffs, samples)
        for i in range(start_index, min(start_index + len(diffs), total_polls)):
            initial = reference_diffs[i]
            reference_diffs[i] = diffs[i-start_index] + initial
        #print(reference_diffs)
        # add reference to diffs
    #print(reference_diffs)
    # trim to length of audio-buffer to compare
    #print("Len of dynamically gererated reference {}".format(len(reference_diffs)))
    #if offset != 0:
    #    return reference_diffs[-reference_polls:]
    #else: 
    #    return reference_diffs
    return reference_diffs[-reference_polls:]

def main(args):
    #test = generate_clean_file(10, "test")
    #hallo_ref = "referenceData/referenceSound/lars_test_hallo_action_0.05.wav"
    #first = add_reference(test, hallo_ref, 4)
    #second = add_reference(first, hallo_ref, 5)
    #prepend_silence(7, "referenceData/referenceSound/lars_test_hallo_action_0.05.wav")
    model_db = load_model_db(referenceDB)
    events = []
    #events.append({"device": "lars","netFN":"test", "command_bytes": "hallo" ,"record_until": time.time_ns()+ 5*1000000000, "ts": time.time_ns()})
    #events.append({"device": "sebastian","netFN":"test", "command_bytes": "hallo" ,"record_until": time.time_ns()+ 5*1000000000, "ts": time.time_ns()})
    #events.append({"device": "lars","netFN":"test", "command_bytes": "hallo" ,"record_until": time.time_ns()+ 5*1000000000, "ts": time.time_ns()+3000000000})
    #test = generate_dynamic_reference(events, time.time_ns()-1000000000, 800, model_db)
    #print(test)
    #trim_reference(test, 2.2, 7.2)
    #(y, sr) = read_file("./data/sound/power_on3.wav")
    #test_rms_extraction = generate_rms_command_reference(y, sr)
    #(y, sr) = read_file("./data/sound/power_off2.wav")
    #test_rms_extraction = generate_rms_command_reference(y, sr)
    test = sox_generate_clean_file(8.0)
    (s_diffs, s_times) = get_rms_change(test)
    start = time.time_ns()
    events = []
    for i in range(2):
        ts = random.randint(start, start+10**9*(10//4))# 2 for each even, generate a random timestamp in the
        print(start) 
        print(ts)
        entry = model_db.sample()
        entry.reset_index(drop=True, inplace=True)
        event = {"ts": ts, "device": entry["device"].to_string(index=False), "netFN": entry["netFN"].to_string(index=False), "command_bytes": entry["command_bytes"].to_string(index=False), "req": True}
        events.append(event) # add event to list
    #plot_rms_diff(s_diffs, s_diffs, s_times)
    print(events)
    r = generate_dynamic_rms_reference(events, time.time_ns(), 8, model_db, len(s_diffs), 2)
    reference_rms_signal = []
    s = 0
    for i in range(len(r)):
        s += r[i]
        reference_rms_signal.append(s)
    plot_rms_diff(reference_rms_signal,r, s_times)
    pass

if __name__=="__main__":
    main(sys.argv)
