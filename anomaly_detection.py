# ! /usr/bin/python3

import configparser
import asyncio
import statistics
import sox
import json
import sys
import time
import os
import shutil
import datetime
from threading import Thread
from validate import validate, validate_cnn
from modeldb_builder import init_modeldb
from network_traffic.ipmi_processor import process_live_traffic
from sidechannel.audio.sound_recording import AudioHandler, store_audio, read_file, sd_Audiohandler
from sidechannel.audio.dynamic_reference_creation import generate_dynamic_reference, load_model_db, check_reference
from sidechannel.audio.reference_building import get_action_length
from sidechannel.audio.sound_module.signal_analysis import limit_frequencies
from sidechannel.audio.sound_module.plotting import plot_spectrogram, plot_double_spectrogram
from event_list import events

statistics = {"start": None, "end": None, "polls": [], "events": [], "duration": None}


def validate_buffer(buffer_end, audio, events, length, offset, modeldb, model_db_dir, noise_prof=None):
    """
        Input:
            microphone: audioinput where the buffer can be pulled from
            events: list of current event list
            length: length of audiobuffer (i.e. how long the audio is that should be compared) (seconds)
            offset: time between reference start and audio start (alignment) (seconds)
            modeldb: model-db dataframe
            model-db-dir: location where reference files are located
    """
    # filter to frequency spectrum
    limited_audio = limit_frequencies(audio, 44100, 400, 700)
    # store audio in file
    audio_recording = store_audio(limited_audio, 44100, "timeframe_{}".format(buffer_end))
    ref_start = buffer_end - offset*10**9 - length*10**9  # goal in NS
    validation_start = buffer_end - length*10**9
    # TODO Denoise amount
    if noise_prof is not None:
        cleaned_file = "{}_cleaned_{}.wav".format(audio_recording.split(".wav")[0], 0.05)
        tfm = sox.Transformer()
        tfm.noisered(profile_path=noise_prof, amount=0.08)
        tfm.build(input_filepath="{}".format(audio_recording), output_filepath=cleaned_file)
        os.replace(cleaned_file, audio_recording)
    # get relevant events
    ref_events = []
    for e in events:
        if not e["req"]:  # only consider requests
            continue
        if e["ts"] > buffer_end or e["ts"] < ref_start:  # if an event starts before the start-time of the reference or after the end of the recording
            continue
        else:
            if len(check_reference(e, modeldb)) > 0:
                ref_events.append(e)
    # print(ref_events)
    ref_events = []
    debug = True
    if not ref_events and not debug:
        # no active events
        # NOMS to recording
        # return early during testing other function
        print("no relevant events")
        # os.remove(audio_recording)
        # return False
        meta = get_action_length(audio_recording, viz=True)
        os.remove(audio_recording)
        if meta[1] == -1 or meta[2] == -1:
            # no start found => no activity -> good
            print("Timeframe from {} to {} validated".format(
                datetime.datetime.fromtimestamp(validation_start/1e9).strftime('%c'), datetime.datetime.fromtimestamp(buffer_end/1e9).strftime('%c')))
            return True
        else:
            # there was some activity: bad
            print("WARNING: Recording from {} to {} did not match the expectation".format(
                datetime.datetime.fromtimestamp(validation_start/1e9).strftime('%c'), datetime.datetime.fromtimestamp(buffer_end/1e9).strftime('%c')))
            return False
    # DEBUG STORE
    if not ref_events:
        print("Debugging")
    shutil.copyfile(audio_recording, "anomaly_audio_test.wav")
    # if there are events, validate with fingerprinting
    reference = generate_dynamic_reference(ref_events, ref_start, length, modeldb, model_db_dir, offset)  # times of len and offset in 10^(-2)s
    # plot_double_spectrogram(read_file(audio_recording)[0], read_file(reference)[0], "anomaly_detection")
    reference_dir = ("./{}".format(buffer_end))
    os.mkdir(reference_dir)
    os.replace(reference, "{}/{}".format(reference_dir, reference))
    ## os.replace(reference, "{}/{}".format(reference_dir, audio_recording))
    # generate directory and move reference to it
    # store audiobuffer to file
    # compare files for validation
    # similarity = validate(reference_dir, audio_recording, 0.3)
    print("entry CNN check")
    result_class = validate_cnn(audio_recording, f"{reference_dir}/{reference}")
    print(f"CNN RESULT: {result_class}")
    if result_class == "normal":
        similarity = True
    else:
        similarity = False
    ## similarity = validate(reference_dir, reference, 0.3)
    # cleanup
    ## cleanup reference file
    shutil.rmtree(reference_dir)
    ## cleanup recording
    os.remove(audio_recording)
    # print validated or not
    if similarity:
        print("Timeframe from {} to {} validated".format(
            datetime.datetime.fromtimestamp(validation_start/1e9).strftime('%c'), datetime.datetime.fromtimestamp(buffer_end/1e9).strftime('%c')))
        return True
    else:
        print("WARNING: Recording from {} to {} did not match the expectation".format(
            datetime.datetime.fromtimestamp(validation_start/1e9).strftime('%c'), datetime.datetime.fromtimestamp(buffer_end/1e9).strftime('%c')))
        return False


# from https://stackoverflow.com/questions/69939800/run-a-function-every-n-seconds-in-python-with-asyncio
async def ad_loop(__seconds: float, mic, config, modeldb, modeldb_dir):
    if config["DATASTORAGE"]["store"] == "True":
        store = True
        data_collection = config["DATASTORAGE"]["soundDir"]
    else:
        store = False
    while True:
        (end, audio_data) = mic.get_buffer()
        event_copy = events.copy()
        poll = {"end": end, "result": None, "events": [], "start": None}
        start = end - int(config["DETECTION"]["DetectionAudioSemiknown"])*10**9 - int(config["DETECTION"]["DetectionAudioBufferLength"])*10**9
        poll["start"] = start
        for e in events:
            if not e["req"]:
                # only consider requests
                continue
            if e["ts"] > end or e["ts"] < start:  # if an event starts before the start-time of the reference or after the end of the recording
                continue
            else:
                if len(check_reference(e, modeldb)) > 0:
                    poll["events"].append(e)
        poll["result"] = validate_buffer(end, audio_data, event_copy, int(config["DETECTION"]["DetectionAudioBufferLength"]), int(config["DETECTION"]["DetectionAudioSemiknown"]), modeldb, modeldb_dir, config["DETECTION"]["noiseprofile"])
        # validate_buffer(end, audio_data, event_copy, int(config["DETECTION"]["DetectionAudioBufferLength"]), int(config["DETECTION"]["DetectionAudioSemiknown"]), modeldb, modeldb_dir)
        if store:
            try:
                f_name = f"{data_collection}/{end - int(config['DETECTION']['DetectionAudioBufferLength'])*10**9}_ad"  # end of the ts but the label of the audio file should indicate the beginning
                storer = Thread(target=store_audio, args=[audio_data, mic.RATE, f_name])
                storer.start()
                print(f"Stored audio in {f_name}.wav")
            except:
                print("Failed to store the buffer")
        statistics["polls"].append(poll)
        await asyncio.sleep(__seconds)


async def ad_loop_experiment(__seconds: float, mic, config, modeldb, modeldb_dir, duration=30):
    exp_end = statistics["start"] + datetime.timedelta(0, duration)
    print("Experiment started at: {}\t Duration: {}s \t End at: {}".format(statistics["start"], duration, exp_end))
    if config["DATASTORAGE"]["store"] == "True":
        store = True
        data_collection = config["DATASTORAGE"]["soundDir"]
    else:
        store = False
    while datetime.datetime.now() < exp_end:
        (end, audio_data) = mic.get_buffer()
        event_copy = events.copy()
        poll = {"end": end, "result": None, "events": [], "start": None}
        start = end - int(config["DETECTION"]["DetectionAudioSemiknown"])*10**9 - int(config["DETECTION"]["DetectionAudioBufferLength"])*10**9
        poll["start"] = start
        for e in events:
            if not e["req"]:
                # only consider requests
                continue
            if e["ts"] > end or e["ts"] < start:  # if an event starts before the start-time of the reference or after the end of the recording
                continue
            else:
                if len(check_reference(e, modeldb)) > 0:
                    poll["events"].append(e)
        poll["result"] = validate_buffer(end, audio_data, event_copy, int(config["DETECTION"]["DetectionAudioBufferLength"]), int(config["DETECTION"]["DetectionAudioSemiknown"]), modeldb, modeldb_dir, config["DETECTION"]["noiseprofile"])
        # validate_buffer(end, audio_data, event_copy, int(config["DETECTION"]["DetectionAudioBufferLength"]), int(config["DETECTION"]["DetectionAudioSemiknown"]), modeldb, modeldb_dir)
        statistics["polls"].append(poll)
        if store:
            try:
                f_name = f"{data_collection}/{end - int(config['DETECTION']['DetectionAudioBufferLength'])*10**9}_ad"
                storer = Thread(target=store_audio, args=[audio_data, mic.RATE, f_name])
                storer.start()
                print(f"Stored audio in {f_name}.wav")
            except:
                print("Failed to store the buffer")
        await asyncio.sleep(__seconds)
    statistics["end"] = datetime.datetime.now()
    statistics["duration"] = statistics["end"] - statistics["start"]
    with open("experiment_results.json", "w") as f:
        json.dump(statistics, f, default=str)
    print("Experiment finished, data stored in experiment_results.json")


def main(args):
    # init the db and get data from config
    config = configparser.ConfigParser()
    config.read("config.ini")
    interface = config["INTERFACES"]["interface"]
    password = config["INTERFACES"]["ipmi_pass"].encode("utf-8")
    modeldb_path = init_modeldb(config["MODELDB"]["ModelDBDir"])
    modeldb = load_model_db(modeldb_path)
    modeldb_dir = config["MODELDB"]["ModelDBDir"]

    # start interface capture and microphone
    interface_capture = Thread(target=process_live_traffic, args=[interface, password])
    interface_capture.start()
    # mic = AudioHandler(int(config["DETECTION"]["DetectionAudioBufferLength"])) # 10 second audio-ringbuffer
    mic = sd_Audiohandler(int(config["DETECTION"]["DetectionAudioBufferLength"]))
    print("Length of audiobuffer: {}".format(int(config["DETECTION"]["DetectionAudioBufferLength"])))
    # mic.start()
    # microphone_processor = Thread(target=mic.mainloop)
    # microphone_processor.start()
    # loop every n-1 seconds (n= length of audiobuffer)
    with mic.start():
        print("loaded interfaced, building buffer")
        time.sleep(15)
        statistics["start"] = datetime.datetime.now()
        print("Starting Anomaly Detection")
        a = asyncio.get_event_loop()
        a.create_task(ad_loop_experiment(int(config["DETECTION"]["DetectionAudioBufferLength"])-3, mic, config, modeldb, modeldb_dir, 21))
        a.run_forever()
        # (end, audio_data) = mic.get_buffer()
        # validate_buffer(end, audio_data, events, int(config["DETECTION"]["DetectionAudioBufferLength"]), int(config["DETECTION"]["DetectionAudioSemiknown"]), modeldb, modeldb_dir)


if __name__=="__main__":
    main(sys.argv)
