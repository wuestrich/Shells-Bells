
import librosa
import librosa.display
import sys
import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from sound_recording import read_file

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

test_csv_file = "./test_ml_poweron.csv"


def generate_sound_video(results:list):
    #### plot the peaks 
    folder = "./sound_videos"
    frames = []
    for i,result in enumerate(results):
        plt.figure()
        plt.plot(result)
        plt.plot([0.145]*len(result),label="0.145")
        plt.plot([np.average(result)]*len(result), label="average")
        plt.ylim(0.0,2.0)
        plt.legend()
        plt.savefig(folder+"/file%02d.png" %i)
        plt.close()
    subprocess.call(['ffmpeg', '-framerate', '8', '-i', f'{folder}/file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'sound_video.mp4'])


def analyze_timeframe(y, frequency_split=3):
    """ frequency bands indicates how the complete band should be split """
    time_x_frequencies = np.abs(librosa.stft(y, n_fft=2048))  # abs to remove complex parts, change n_fft for more or less accuracy
    frequencies_x_times = np.transpose(time_x_frequencies)
    band_average_aggregate = [[]]*frequency_split
    for frequencies in frequencies_x_times:  # every iteration is all frequencies during a time window
        frequency_bands = np.array_split(frequencies, 4) # split into four frequency bands
        for i, band in enumerate(frequency_bands):
            buckets = np.array_split(band, 128)  # split frequency band in to buckets, 128: number of buckets
            bucket_aggregation = [np.average(bucket) for bucket in buckets]
            bucket_amplification = [min(2, (x) ** 3) for x in bucket_aggregation]
            band_average_aggregate[i].append(bucket_amplification)
    averages = [np.array(band).mean(axis=0) for band in band_average_aggregate]
    band_deviations = [[]]*frequency_split
    for i, band in enumerate(band_average_aggregate):
        for j, time_frame in enumerate(band):
            cleaned = []
            for k in range(len(time_frame)):
                cleaned.append(abs(time_frame[k]-3*averages[i][k]))
            band_deviations[i].append(cleaned)
    
    transposed = []
    for band in band_deviations:
        transposed.append(np.transpose(band))
    significant_changes = [[]]*frequency_split
    for j, band in enumerate(transposed):
        for i, frame in enumerate(band):
            f_avg = np.average(frame)
            res = [1 if x > f_avg * 2.5 else 0 for x in frame]
            # res = [1 if x > win_avg * 5 else 0 for x in frame]
            significant_changes[j].append(res)

    for band in significant_changes:
        librosa.display.specshow(librosa.amplitude_to_db(band, ref=np.max),y_axis="linear", x_axis="time")
        plt.show()
    pass


def remove_small_artifacts(bitmap, window_x=1, window_y=1, min_activation=1):
    """Remove all artifacts from a bitmap which do not have many active neighbors in proximity"""
    filtered_bitmap = []
    for x, freq in enumerate(bitmap):
        frequency= []
        for y in range(0, len(bitmap[x])):
            if bitmap[x][y] == 0.0:
                frequency.append(0.0)
            elif bitmap[x][y] == 1.0:
                if x > window_x-1 and x < (len(bitmap)-window_x -1) and y > 0 and y < (len(bitmap[x])-window_y-1):
                    activation = 0
                    for i in range(max(x-window_x-1,0),min(x+window_x-1, len(bitmap))):
                        for j in range(max(y-window_y-1, 0), min(y+window_y-1, len(bitmap[x]))):
                            activation += bitmap[i][j]
                    if activation >= min_activation:
                        frequency.append(1.0)
                    else:
                        frequency.append(0.0)
                else:
                    frequency.append(bitmap[x][y])
        filtered_bitmap.append(frequency)
    return filtered_bitmap


def highlight_significant_changes(f_name, viz=False, amplification=3, avg_dist=3, max_ceil=2, abs_diff_factor=1.47):
    (y, sr) = read_file(f_name)
    time_x_frequencies = np.abs(librosa.stft(y, n_fft=2048)) 
    #if viz:
    #    img = librosa.display.specshow(librosa.amplitude_to_db(time_x_frequencies, ref=np.max),y_axis="linear", x_axis="time")#, sr=44100)
    #    plt.show()
    frequencies_x_times = np.transpose(time_x_frequencies)
    averaged_results = []
    for timeslice in frequencies_x_times:
        frequency_bucket = np.array_split(timeslice[:len(timeslice)//4], 128)  # len(frame)//4: only consider lowest quarter of frequencies of the full spectrum, 128: number of bins
        # frequency_bucket = np.array_split(timeslice, 512)
        # bucket_values = [np.max(x) for x in frequency_bucket]
        bucket_values = [np.average(x) for x in frequency_bucket]
        # representative_amplification = [(x) ** 3 for x in bucket_values]
        representative_amplification = [min(max_ceil, (x) ** amplification) for x in bucket_values] # min is for the ceiling, may need to be adapted for other things
        averaged_results.append(representative_amplification)
    # r_max = np.max(results)
    # r_min = np.min(results)
    # print(r_min)
    # generate_sound_video(results)
    ## subtract the average value to get the major changes
    results = []
    averages = np.array(averaged_results).mean(axis=0)
    for timeslice in averaged_results:
        cleaned = []
        for i in range(len(timeslice)):
            cleaned.append(abs(max(0,timeslice[i]-avg_dist*averages[i]))) # default 3
        results.append(cleaned)
    #results = averaged_results
    results = np.transpose(results)
    #np.savetxt("ref.csv", results, delimiter=",")
    res_final = []
    # dirty hack for data generation: if some areas are below x amount of max, then don't consider them as well
    r_max = np.amax(results)
    for i, frequency in enumerate(results):
        f_avg = np.average(frequency)
        res = [1.0 if x > f_avg * abs_diff_factor and x > r_max/100 else 0.0 for x in frequency] #f_avg * 1.47 yields good results for the complete recording, 5 for single events
        res_final.append(res)
    res_final_filtered = remove_small_artifacts(res_final, 2, 4, 7) # 2,2,3 yields good results for the complete recording, 3.,3, 8 for constructed
    res_final = np.array(res_final_filtered)
    if viz:
        img = librosa.display.specshow(librosa.amplitude_to_db(res_final, ref=np.max),y_axis="linear", x_axis="time")#, sr=44100)
        plt.show()
        t2 = results
        img = librosa.display.specshow(librosa.amplitude_to_db(t2, ref=np.max),y_axis="linear", x_axis="time")#, sr=44100)
        plt.show()
    return res_final


def highlight_significant_changes_visualize_steps(f_name):
    (y, sr) = read_file(f_name)
    signal_stft = librosa.stft(y, n_fft=2048)
    print("Processed signal: Input signal spectrogram")
    img = librosa.display.specshow(librosa.amplitude_to_db(signal_stft, ref=np.max),y_axis="linear", x_axis="time")#, sr=44100)
    plt.show()
    print("Removed imaginary parts")
    time_x_frequencies = np.abs(librosa.stft(y, n_fft=2048))
    print("Spectrogram of input signal") 
    img = librosa.display.specshow(librosa.amplitude_to_db(time_x_frequencies, ref=np.max),y_axis="linear", x_axis="time")#, sr=44100)
    plt.show()
    print("Continuing, limiting frequency band")
    frequencies_x_times = np.transpose(time_x_frequencies)
    frequency_limitation = []
    for timeslice in frequencies_x_times:
        frequency_limitation.append(timeslice[:len(timeslice)//4])
    print("Frequency Band limited, current visualization of spectrogram")
    img = librosa.display.specshow(librosa.amplitude_to_db(np.transpose(frequency_limitation), ref=np.max),y_axis="linear", x_axis="time")
    plt.show()
    intermediate = []
    for timeslice in frequencies_x_times:
        frequency_bucket = np.array_split(timeslice[:len(timeslice)//4], 128)
        bucket_values = [np.average(x) for x in frequency_bucket]
        intermediate.append(bucket_values)
    print("Created average of buckets")
    img = librosa.display.specshow(librosa.amplitude_to_db(np.transpose(intermediate), ref=np.max),y_axis="linear", x_axis="time")
    plt.show()
    averaged_results = []
    for timeslice in intermediate:
        representative_amplification = [min(2, (x) ** 5) for x in timeslice] # original recoding exp 3
        averaged_results.append(representative_amplification)
    print("Amplification Step Visualization:")
    img = librosa.display.specshow(librosa.amplitude_to_db(np.transpose(averaged_results), ref=np.max),y_axis="linear", x_axis="time")
    plt.show()
    results = []
    averages = np.array(averaged_results).mean(axis=0)
    for timeslice in averaged_results:
        cleaned = []
        for i in range(len(timeslice)):
            cleaned.append(abs(timeslice[i]-3*averages[i]))
        results.append(cleaned)
    print("Subtracted average from frequencies")
    results = np.transpose(results)
    res_final = []
    for i, frequency in enumerate(results):
        f_avg = np.average(frequency)
        res = [1.0 if x > f_avg * 1.47 else 0.0 for x in frequency] #f_avg * 1.47 yields good results for the complete recording
        res_final.append(res)
    print("Initial bitmap with outliers")
    img = librosa.display.specshow(librosa.amplitude_to_db(res_final, ref=np.max),y_axis="linear", x_axis="time")
    plt.show()
    res_final_filtered = remove_small_artifacts(res_final, 2, 2, 3) # 2,2,3 yields good results for the complete recording
    print("removed single outliers")
    img = librosa.display.specshow(librosa.amplitude_to_db(res_final_filtered, ref=np.max),y_axis="linear", x_axis="time")  #, sr=44100)
    plt.show()


def find_config(f_name):
    for amp in range(2,5):
        for avg_dist in np.arange(2, 4, 0.5):
            for min_dist_factor in np.arange(1.3, 2.1, 0.1):
                highlight_significant_changes(f_name=f_name, amplification=amp, avg_dist=avg_dist, abs_diff_factor=min_dist_factor, viz=True)
                print(f"Amplification: {amp}\t, Avg distance subtraction: {avg_dist}, Min distance to average factor {min_dist_factor}")


def main(args):
    pass


if __name__ == "__main__":
    main(sys.argv)