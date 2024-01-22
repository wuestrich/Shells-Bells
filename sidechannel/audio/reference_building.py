#! /usr/bin/env python3.8
# coding: utf-8

import sys
import sox
import csv
if __name__ == "__main__":  # if run as standalone
    from sound_module.signal_analysis import limit_frequencies, rms_energy, get_event_start, get_event_end, rms_energy_smoothed
    from sound_module.plotting import plot_spectrogram, plot_spectrogram_rms_smoothrms, plot_rms_diff
    from sound_recording import store_audio, read_file
else:  # import as module can handle relative imports
    from .sound_module.signal_analysis import limit_frequencies, rms_energy, get_event_start, get_event_end, rms_energy_smoothed
    from .sound_module.plotting import plot_spectrogram, plot_spectrogram_rms_smoothrms, plot_rms_diff
    from .sound_recording import store_audio, read_file


# from testing.test_verification import generate_random_reference

referenceDB = ""
s_to_ns = 1e9


def generate_command_reference(sound, event, sr, end_ts, reference_path=""):
    """Generates a reference from a detected event and an accompanying sound"""
    # TODO Refactor audio-name, this is way too complex at the moment
    reference = {}
    reference["device"] = event["device"]
    reference["netFN"] = event["netFN"]
    reference["command_bytes"] = event["command_bytes"]
    f_name = "{}_{}_{}".format(reference["device"], reference["netFN"], reference["command_bytes"])
    # 1. store sound recording to file
    audio_f_name = "{}/referenceSound/{}".format(reference_path, f_name)  # relevant suffixes that have the soud _action_{}.wav, _start_noise_profile, .wav,
    # filtering for fans
    filtered_audio = limit_frequencies(sound, sr, 400, 900)
    in_audio = store_audio(filtered_audio, sr, audio_f_name)
    # plot_spectrogram(read_file(in_audio)[1],f_name="reference_building")
    # 2. determine length of action
    meta = get_action_length(in_audio, True)  # meta[0]=duration, meta[1]= start_ts (relative from beginning of recording) meta[2]=end_ts (relative from beginning of recording)
    if meta[1] == -1 or meta[2] == -1:
        print(meta)
        print("ERROR: Failed to determine length of action")
        if meta[1] == -1:
            print('ERROR: No start identfied')
        else:
            print('ERROR: No end identified')
        return None
    # 3. generate cleaned reference of action
    reference["duration"] = meta[0]
    beginnig = extract_start(meta[1], in_audio)
    noise_prof = build_noise_profile(beginnig)
    tfm = sox.Transformer()
    tfm.noisered(profile_path=noise_prof, amount=0.12)
    tfm.trim(meta[1], meta[2])  # split remove sound before and after action
    cleaned_file = "{}_action_{}.wav".format(audio_f_name.split(".wav")[0], 0.05)
    tfm.build(input_filepath="{}.wav".format(audio_f_name), output_filepath=cleaned_file)
    # 4. calculate from command to time to action (tta)
    # tta = action_start_ts - command_ts
    # action_start_ts = end_ts - len(recording) + start_action
    action_start_ts = end_ts - ((len(sound)/sr)*s_to_ns) + meta[1]*s_to_ns
    reference["tta"] = (action_start_ts - event["ts"]) / s_to_ns
    reference["sound_file_prefix"] = audio_f_name
    # store to csv
    with open("{}/modeldb.csv".format(reference_path), "a") as f:
        w = csv.DictWriter(f, reference.keys())
        # w.writeheader()
        w.writerow(reference)
    return 1


def generate_rms_command_reference(sound, event, sr, end_ts, reference_path=""):
    reference = {}
    reference["device"] = event["device"]
    reference["netFN"] = event["netFN"]
    reference["command_bytes"] = event["command_bytes"]
    f_name = "{}_{}_{}".format(reference["device"], reference["netFN"], reference["command_bytes"])
    audio_f_name = "{}/referenceSound/{}".format(reference_path, f_name)  # relevant suffixes that have the sound _action_{}.wav, _start_noise_profile, .wav,
    filtered_audio = limit_frequencies(sound, sr, 400, 900)
    in_audio = store_audio(filtered_audio, sr, audio_f_name)
    meta = get_action_length(in_audio, False)  # meta[0]=duration, meta[1]= start_ts (relative from beginning of recording) meta[2]=end_ts (relative from beginning of recording)
    if meta[1] == -1 or meta[2] == -1:
        print(meta)
        print("ERROR: Failed to determine length of action")
        if meta[1] == -1:
            print('ERROR: No start identfied')
        else:
            print('ERROR: No end identified')
        return None
    reference["duration"] = meta[0]
    (diffs, diff_times) = get_rms_change(in_audio, meta[1], meta[2])
    # 4. generate cleaned reference of action
    beginnig = extract_start(meta[1], in_audio)
    noise_prof = build_noise_profile(beginnig)
    tfm = sox.Transformer()
    tfm.noisered(profile_path=noise_prof, amount=0.12)
    tfm.trim(meta[1], meta[2])  # split remove sound before and after action
    cleaned_file = "{}_action_{}.wav".format(audio_f_name.split(".wav")[0], 0.05)
    tfm.build(input_filepath="{}.wav".format(audio_f_name), output_filepath=cleaned_file)
    # 4. calculate from command to time to action (tta)
    # tta = action_start_ts - command_ts
    # action_start_ts = end_ts - len(recording) + start_action
    action_start_ts = end_ts - ((len(sound)/sr)*s_to_ns) + meta[1]*s_to_ns
    reference["tta"] = (action_start_ts - event["ts"]) / s_to_ns
    reference["sound_file_prefix"] = audio_f_name
    reference["rms_diffs"] = diffs
    reference["rms_diff_times"] = diff_times
    # store to csv
    with open("{}/modeldb.csv".format(reference_path), "a") as f:
        w = csv.DictWriter(f, reference.keys())
        # w.writeheader()
        w.writerow(reference)


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
    (S, rms, times) = rms_energy(y, sr)
    # windows and threshhold selected for fan spinup
    (avg_rms, avg_rms_times) = rms_energy_smoothed(rms, times, n=60)
    (start, start_time) = get_event_start(avg_rms, avg_rms_times, thresh=0.0007, w=60)
    (end, end_time) = get_event_end(avg_rms, avg_rms_times, start, thresh=0.0004, w=50)
    duration = end_time-start_time
    if viz:
        plot_spectrogram_rms_smoothrms(S, rms, times, avg_rms, avg_rms_times, f_name="reference_building")
    return (duration, start_time, end_time)


def get_rms_change(f_name, start=None, end=None, viz=False):
    """Returns the RMS change due to activity in the input file
    input:
        f_name: filename from which the diff should be extracted
        start: optional: if set specify start timestamp for analysis
        end: optional if set specifies end timestamp for analysis
    output
        (rms_diff, rms_diff_times): differences to base signal, and timestamps at which these happen"""
    (y, sr) = read_file(f_name)
    (S, rms, times) = rms_energy(y, sr)
    (avg_rms, avg_rms_times) = rms_energy_smoothed(rms, times, n=60)
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
    # y = signal_analysis.limit_frequencies(y, sr, 256,1700)
    # sf.write("{}_filtered.wav".format(f_name.split(".wav")[0]), y, sr, subtype="PCM_24")
    # (y, sr) = signal_analysis.read_file("{}_filtered.wav".format(f_name.split(".wav")[0]))
    #### End Pre-bandpass filter
    # Step 2: write stuff happening before an actual action a file
    start_file = extract_start(start, f_name)
    # Step 3: calculate noise profile from file
    noise_prof = build_noise_profile(start_file)
    # Step 4: denoise initial file with noise profile
    tfm = sox.Transformer()
    # loop to determine best values for sox denoise
    tfm.noisered(profile_path=noise_prof, amount=0.12)
    cleaned_file = "{}_denoised_{}.wav".format(f_name.split(".wav")[0], 0.12)
    tfm.build(input_filepath=f_name, output_filepath=cleaned_file)


def extract_start(time, f_name):
    """Writes the start of a file to a new file up to a specified timestamp
    Input:
        f_name: filename of recording from which the beginning should be extracted
        time: timestamp until which the recoring should take place
    Output:
        out_f_name: filename of output file"""
    tfm = sox.Transformer()
    tfm.trim(0.0, time)  # time -1 just to mitigate potential slow detection beginning
    out_file = "{}_start.wav".format(f_name.split(".wav")[0])
    tfm.build(input_filepath=f_name, output_filepath=out_file)
    return out_file


def build_noise_profile(f_name):
    tfm = sox.Transformer()
    profile_fpath = "{}_noiseprofile".format(f_name.split(".wav")[0])
    tfm.noiseprof(input_filepath=f_name, profile_path=profile_fpath)
    return profile_fpath


def extract_action(f_name):
    """given a soundfile, extract the action timeframe with the NOMS algorithm"""
    # TODO Refactor audio-name, this is way too complex at the moment
    # 1. store sound recording to file
    audio_f_name = f"{f_name}_extracted"  # relevant suffixes that have the sound _action_{}.wav, _start_noise_profile, .wav,
    sound = read_file(f_name)
    sr = 44100  # in shells bells SR is always 44100
    # filtering for fans
    filtered_audio = limit_frequencies(sound, sr, 400, 900)
    in_audio = store_audio(filtered_audio, sr, audio_f_name)
    # plot_spectrogram(read_file(in_audio)[1],f_name="reference_building")
    # 2. determine length of action
    meta = get_action_length(in_audio, True)  # meta[0]=duration, meta[1]= start_ts (relative from beginning of recording) meta[2]=end_ts (relative from beginning of recording)
    if meta[1] == -1 or meta[2] == -1:
        print(meta)
        print("ERROR: Failed to determine length of action")
        return None
    # 3. generate cleaned reference of action
    # reference["duration"]=meta[0]
    beginnig = extract_start(meta[1], in_audio)
    noise_prof = build_noise_profile(beginnig)
    filter_amount = 0.12
    tfm = sox.Transformer()
    tfm.noisered(profile_path=noise_prof, amount=filter_amount)
    tfm.trim(meta[1], meta[2])  # split remove sound before and after action
    cleaned_file = "{}_action_{}.wav".format(audio_f_name.split(".wav")[0], filter_amount)
    tfm.build(input_filepath="{}.wav".format(audio_f_name), output_filepath=cleaned_file)
    # 4. calculate from command to time to action (tta)
    # tta = action_start_ts - command_ts
    # action_start_ts = end_ts - len(recording) + start_action
    # action_start_ts = end_ts - ((len(sound)/sr)*s_to_ns) + meta[1]*s_to_ns
    # reference["tta"] = (action_start_ts - event["ts"]) / s_to_ns
    # reference["sound_file_prefix"] = audio_f_name
    # store to csv
    # with open("{}/modeldb.csv".format(reference_path), "a") as f:
    #    w = csv.DictWriter(f, reference.keys())
    #    # w.writeheader()
    #    w.writerow(reference)
    # print(reference)
    ##TODO include response code in reference
    return 1


def main(args):
    pass


if __name__ == "__main__":
    main(sys.argv)
