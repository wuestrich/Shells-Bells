#! /usr/bin/env python3.7
# coding: utf-8

import subprocess
import ast
import random
import sys
import sox
import string
import pandas as pd
import time
import os
if __name__ == "__main__":  # if run as standalone
    from sound_recording import store_audio, read_file
    # import sound_module.signal_analysis as signal_analysis
    from sound_module.plotting import plot_spectrogram, plot_spectrogram_rms_smoothrms, plot_rms_diff
    from reference_building import get_rms_change
else:  # import as module can handle relative imports
    from .sound_recording import store_audio, read_file
    # import sound_module.signal_analysis as signal_analysis
    from .sound_module.plotting import plot_spectrogram, plot_spectrogram_rms_smoothrms, plot_rms_diff
    from .reference_building import get_rms_change
# from testing.test_verification import generate_random_reference

referenceDB = ""


def sox_generate_clean_file(period, f_name="silence"):
    command = ["sox", "-n", "-r", "44100", "{}.wav".format(f_name), "trim", "0.0", "{}".format(period)]
    ret = subprocess.run(command, capture_output=False)
    return "{}.wav".format(f_name)


def sox_prepend_silence(period, sound):
    """Prepends silent part before a sound
    Input:
        period: length of silence before the sound, sample rate of 44100
            unity of period is s, e.g. 10.2
        sound: path to sound file name to which the silent part should be prepended"""
    silent_prefix = sox_generate_clean_file(period, ''.join(random.choice(string.ascii_lowercase) for x in range(10)))
    tfm = sox.Combiner()
    rn = ''.join(random.choice(string.ascii_lowercase) for x in range(10))
    ref_name = "{}.wav".format(rn)
    tfm.build(input_filepath_list=[silent_prefix, sound], output_filepath=ref_name, combine_type="concatenate")
    os.remove(silent_prefix)
    return ref_name


def generate_clean_file(period, f_name="silence"):
    """Generates a silent sound file that has length of period
    Suggestion: len(buffer)+max(tta+duration) for any action in the reference set
    Input:
        period: time in 10^-2 seconds"""
    out = store_audio([0.0]*441*period, 44100, f_name)
    return out


def prepend_silence(period, sound):
    """Prepends silent part before a sound
    Input:
        period: length of silence before the sound, sample rate of 44100
            unity of period is 10^-2s = cs
        sound: path to sound file name to which the silent part should be prepended"""
    silent_prefix = generate_clean_file(period)
    tfm = sox.Combiner()
    ref_name = "silent_prefix_ref.wav"
    tfm.build(input_filepath_list=[silent_prefix, sound], output_filepath=ref_name, combine_type="concatenate")
    return ref_name


def sox_add_reference(base_file, reference_to_add, ts):
    """ add a sound to an existing sound file at a given timestamp (relative from beginning of a file)"""
    ref_at_ts = sox_prepend_silence(ts, reference_to_add)
    tfm = sox.Combiner()
    out = "ref_{}.wav".format(ts)
    tfm.build(input_filepath_list=[base_file, ref_at_ts], output_filepath=out, combine_type="mix")
    os.remove(ref_at_ts)
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
    tfm.build(input_filepath=reference_file, output_filepath=out)
    os.replace(out, reference_file)
    return reference_file


def load_model_db(model_db_csv: str):
    """loads a model-db from a csv"""
    model_db = pd.read_csv(model_db_csv)
    return model_db


def check_reference(event, model_db):
    """Checks if a given event is in a given model-db and return index of matching row"""
    return model_db.index[(model_db["device"] == event["device"]) & (model_db["netFN"] == event["netFN"]) & (model_db["command_bytes"] == event["command_bytes"])].tolist()


def generate_dynamic_reference(events, start_time, length, model_db, model_db_dir="./", offset=0, name=None):
    """Generates a complete reference for a time-window
    Input:
        events: list of events that should be included in the reference
        start_time: start-time of the reference
        length: length of the reference in s
        offset: if the reference needs to be trimmed. time in 10^-2s until the actual audio in the real world starts
    Output:
        f_name: f_name of file which stores the complete reference"""
    # TODO: cleanup of reference file if something goes wrong to avoid pollution
    total_length = length * 1.0 + offset
    if not name:
        reference = sox_generate_clean_file(total_length, "reference_{}".format(start_time))
    else:
        reference = sox_generate_clean_file(total_length, "reference_{}".format(name))
    for event in events:
        # print("Processing event: ", event)
        ref_check = check_reference(event, model_db)
        if len(ref_check) == 0:  # 1. Check if event is in reference_db,  if not goto next event, if exists:
            print("Event has no reference, continuing")
            continue
        relative_ts = event["ts"] - start_time + int(model_db.loc[ref_check[0]]["tta"]*10**9)  # 2. get ts of command by command_ts - start_time  + tta (convert s to ns)
        # relative ts is in ns, convert+round to 10^-2s due to limited accuracy of sample rate
        rounded_relative_ts = relative_ts * 10 ** -9
        # print("RELATIVE TS OF EVENT: {}".format(rounded_relative_ts))
        ref_file = "{}{}_action_0.05.wav".format(model_db_dir, model_db.loc[ref_check[0]]["sound_file_prefix"])  # get ref
        added = sox_add_reference(reference, ref_file, rounded_relative_ts)  # 3.2 add reference with prepended silence
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
        i = random.randint(0, size-1)
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
        ref_check = check_reference(event, model_db)
        if len(ref_check) == 0:  # 1. Check if event is in reference_db,  if not goto next event, if exists:
            print("Event has no reference, continuing")
            continue
        relative_ts = event["ts"] - start_time + int(model_db.loc[ref_check[0]]["tta"]*10**9)  # 2. get ts of command by command_ts - start_time  + tta (convert s to ns)
        relative_ts_second = relative_ts * 10 ** -9
        event_duration = model_db.loc[ref_check[0]]["duration"]
        # relative ts is in ns, convert+round to 10^-2s = cs due to limited accuracy of sample rate
        event_diffs = ast.literal_eval(model_db.loc[ref_check[0]]["rms_diffs"])
        event_times = ast.literal_eval(model_db.loc[ref_check[0]]["rms_diff_times"])
        # determine start-index
        try:
            start_index = next(x[0] for x in enumerate(reference_diff_times) if x[1] >= relative_ts_second)
        except:
            start_index = len(reference_diff_times)-1
        samples = next(x[0] for x in enumerate(reference_diff_times) if x[1] >= event_duration)
        diffs = rms_resample(event_diffs, samples)
        for i in range(start_index, min(start_index + len(diffs), total_polls)):
            initial = reference_diffs[i]
            reference_diffs[i] = diffs[i-start_index] + initial
        # add reference to diffs
    # trim to length of audio-buffer to compare
    return reference_diffs[-reference_polls:]


def main(args):
    model_db = load_model_db(referenceDB)
    events = []
    test = sox_generate_clean_file(8.0)
    (s_diffs, s_times) = get_rms_change(test)
    start = time.time_ns()
    events = []
    for i in range(2):
        ts = random.randint(start, start+10**9*(10//4))# 2 for each even, generate a random timestamp in the
        entry = model_db.sample()
        entry.reset_index(drop=True, inplace=True)
        event = {"ts": ts, "device": entry["device"].to_string(index=False), "netFN": entry["netFN"].to_string(index=False), "command_bytes": entry["command_bytes"].to_string(index=False), "req": True}
        events.append(event)  # add event to list
    # plot_rms_diff(s_diffs, s_diffs, s_times)
    print(events)
    r = generate_dynamic_rms_reference(events, time.time_ns(), 8, model_db, len(s_diffs), 2)
    reference_rms_signal = []
    s = 0
    for i in range(len(r)):
        s += r[i]
        reference_rms_signal.append(s)
    plot_rms_diff(reference_rms_signal, r, s_times)
    pass


if __name__ == "__main__":
    main(sys.argv)
