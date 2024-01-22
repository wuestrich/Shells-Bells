#! /usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # necessary to import eventlist from the prototype dir
sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pyaudio
import wave
import numpy as np
import time
import sox
import sounddevice as sd
from threading import Thread
import soundfile as sf
if __name__ == "__main__": #if run as standalone
    import ringbuffer
    from sound_module import signal_analysis as signal_analysis
    from sound_module import plotting as plotting
else: # import as module can handle relative imports
    #from . import ringbuffer
    #from .sound_module import signal_analysis as signal_analysis
    #from .sound_module import plotting as plotting
    import ringbuffer
    import sound_module.signal_analysis as signal_analysis

#import ringbuffer
import event_list #TODO fix import
import librosa

class AudioHandler(object):
    # from https://stackoverflow.com/questions/59056786/python-librosa-with-microphone-input
    # https://gist.github.com/mailletf/c49063d005dfc51a2df6
    def __init__(self, buffer_len):
        #self.FORMAT = pyaudio.paFloat32
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 512#1024 #* 2 
        self.p = None
        self.stream = None
        self.ringbuffer = ringbuffer.RingBuffer(buffer_len * self.RATE) # ringbuffer that keeps the last buffer_len seconds
    
    def start(self):
        self.p = pyaudio.PyAudio()
        print("Starting Microphone")
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK,
                                  input_device_index=0
                                  )

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global events
        audio = np.frombuffer(in_data, dtype=np.float32)
        self.ringbuffer.extend(audio)
        #if len(event_list.events) > 0:
        #    print("Network event in buffer")
        #    end_ts = time.time_ns()
        #    if event_list.events[0]["record_until"] < end_ts:
        #        event = event_list.events.pop(0)
        #        buffer_data = self.ringbuffer.get()
        #        print("Processing signal in buffer")
        #        #print(event)
        #        # TODO length of sound capture depending on event, down below fixed 5s of buffer_data
        #        #generate_reference(buffer_data[-(5*44100):], event, self.RATE, end_ts)
        #        generate_command_reference(buffer_data, event, self.RATE, end_ts)
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(0.5)

    def get_buffer(self):
        """
        Returns ringbuffer content and the ts of the ending of the sound data
        Output:
            (ts, ringbuffer_data)
        """
        ts = time.time_ns()
        return (ts, self.ringbuffer.get())


def store_audio(sound, sr, f_name="default"):
    """Stores the provided sound as .wav in a specified directory
    Input: 
        sound: audio-data as frames
        event: event dictionary
        sr: sample-rate of the recording
        f_name: name of file where the audio is supposed to be stored
    Output: 
        f_name.wav: f_name where the sound was stored
        """
    sf.write("{}.wav".format(f_name), sound, sr)
    return "{}.wav".format(f_name)


def read_file(path):
    """
    Reads a file from a given file-path
    Input:
        path: file_path to a sound file for processing
    Output:
        (signal_data, sampling rate)
    """
    samples, sampling_rate = librosa.load(path)
    return (samples, sampling_rate)


def record_n_seconds(seconds, f_name="audio"):
    # adapted from: https://people.csail.mit.edu/hubert/pyaudio/
    # on linux if issues with alsa: https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
    CHUNK = 1024
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = seconds
    WAVE_OUTPUT_FILENAME = "test_audio.wav"
    p = pyaudio.PyAudio()        
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


class sd_Audiohandler(object):
    def __init__(self, buffer_len) -> None:
        self.FORMAT = "float32"
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 512#1024 #* 2 
        self.p = None
        self.stream = None
        self.ringbuffer = ringbuffer.RingBuffer(buffer_len * self.RATE) # ringbuffer that keeps the last buffer_len seconds
    
        pass
    
    def sd_callback(self, indata, frames, time, status):#, status):
        if status:
            print(status)
        self.ringbuffer.extend(indata.copy().squeeze())

    def start(self):
        #duration_chunks = 5.5
        #sd.default.blocksize = 1
        print(f"Sound Stream started on {sd.query_devices(sd.default.device[0])['name']}")
        stream = sd.InputStream(channels=self.CHANNELS, dtype=self.FORMAT, samplerate=self.RATE, callback=self.sd_callback)
        #    #sd.sleep(int(duration_chunks * 1000))
        #    while True: # loop to keep this thing running
        #        continue
        return stream 

    def get_buffer(self):
        """
        Returns ringbuffer content and the ts of the ending of the sound data
        Output:
            (ts, ringbuffer_data)
        """
        ts = time.time_ns()
        return (ts, self.ringbuffer.get())
    
    def stop(self):
        pass
    
    

#ring = ringbuffer.RingBuffer(3 * 44100)



events = []
def main(args):
    ##record_n_seconds(2)
    #audio = AudioHandler(5) # 5 second ringbuffer
    ##for i in range(8):
    ##    print(pyaudio.PyAudio().get_device_info_by_index(i))
    #audio.start()
    ##audio.mainloop()
    #b = Thread(target=audio.mainloop)
    #b.start()
    #print("Filling buffer")
    #time.sleep(5)
    #print("Buffer full")
    #store_audio(audio.get_buffer()[1],44100,"test_audio")
    ##event_list.events.append({"device": "lars","net_fun":"test", "command_bytes": "hallo" ,"record_until": time.time_ns()+ 5*1000000000, "ts": time.time_ns()})
    ##event_list.events.append({"record_until": time.time_ns()+ 6*1000000000})
    ##print(event_list.events)
    ##print("added event, start making some noise")
    ##audio.stop()
    ##sample_denoise("./data/sound/power_on3.wav")
    duration = 5.5
    
    try:
        sd_test = sd_Audiohandler(3)
        with sd_test.start():
            while True:
                continue
    except KeyboardInterrupt:
        print("recording interrupted, storing buffer")
        store_audio(sd_test.get_buffer()[1], 44100, "sd_test")

if __name__=="__main__":
    main(sys.argv)