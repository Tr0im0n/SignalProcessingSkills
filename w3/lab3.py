
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from PIL import Image


def display(signal, sound=False):
    max_amp = np.max(signal)
    signal_norm = signal/max_amp
    plt.plot(signal_norm)
    if sound:
        sd.play(signal_norm)
    plt.title(max_amp)
    plt.show()
    plt.close()


# imported, doesnt work, POG...
def flatten_old(a_list):
    return [item for sublist in a_list for item in sublist]


def flatten(a_list):
    ans = []
    for sublist in a_list:
        for num in sublist:
            ans.append(num)
    return ans


# Exercise ###########################################################################################


def voice1():
    # read the wav file
    file_name = "i_like_apples.wav"
    sr1, data1 = wavfile.read(file_name)
    data2 = np.array([i for i, _ in data1])
    # display(data1)

    # fourier transform the data
    fourier1 = np.fft.rfft(data2)
    fourier2 = abs(fourier1)
    # display(fourier2)

    # pitch the voice upp by rolling and inverse fourier
    shift = 1000
    rolled1 = np.roll(fourier1, shift)
    ifourier1 = np.fft.irfft(rolled1)
    display(ifourier1, True)


def spectrogram1():
    heatmap_list = [[4, 2, 1, 6], [2, 3, 8, 4], [2, 4, 9, 1], [8, 8, 3, 5]]
    plt.imshow(heatmap_list)
    plt.show()


def spectrogram2():
    file_name = "Violin_for_spectrogram.wav"
    sr1, data1 = wavfile.read(file_name)
    sample_len = 441
    wav_len = 261184
    fourier_list = []
    for i in range(0, wav_len-sample_len, sample_len):
        fourier1 = abs(np.fft.rfft(data1[i: i+sample_len]))
        fourier_list.append(fourier1)
    transpose_list = np.transpose(fourier_list)
    plt.imshow(transpose_list, norm="log")
    plt.show()


def lion_to_audio():
    file_name = "lion.png"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)
    print(len(image_array), len(image_array[0]))
    audio_array = []
    for row_data in image_array:
        row_sound = np.fft.irfft(row_data)
        audio_array.append(row_sound)
    full_audio = flatten(audio_array)
    max_amp = np.max(full_audio)
    # min_amp = np.min(full_audio)
    # full_audio1 = np.array(full_audio)
    # full_audio2 = full_audio1 - (1 + min_amp)
    # full_audio3 = full_audio2 / ((max_amp - min_amp) * 2)
    return full_audio / max_amp


def prefix_lion_audio():
    lion_audio = lion_to_audio()
    time_s = 3
    audio_frequency = 44100
    time = np.linspace(0, time_s, time_s*audio_frequency+1)
    freq = 1000
    sine = np.sin(2*np.pi*freq*time)
    return np.append(sine, lion_audio)


def audio_to_lion_old():
    audio = lion_to_audio()
    # print(len(audio))
    rows = 680
    cols = 1066
    image_array = []
    for i in range(rows):
        row_sound = abs( np.fft.rfft(audio[i*cols: (i+1)*cols]) )
        image_array.append(row_sound)
    plt.imshow(image_array)
    plt.show()


def audio_to_lion():
    file_name = "My_recording_3.wav"
    sr1, data1 = wavfile.read(file_name)
    len_data1 = len(data1)
    print(sr1)
    # display(data1)
    audio_frequency = 44100
    sr2 = 16000
    rows = 680
    cols = 1066  # 1066, 387
    start_x = 267400     # 65299, 267400
    end_x = start_x + (rows*cols)
    image_array = []
    for i in range(start_x, len_data1-cols, cols):
        # a_list = data1[i:i+cols]
        row_sound = abs(np.fft.rfft(data1[i:i+cols]))
        # print(row_sound)
        image_array.append(row_sound)
    image_array = np.array(image_array)
    # image_array[image_array > (np.max(image_array) *0.1)] = np.max(image_array) * 0.1
    plt.imshow(image_array)
    plt.show()


# display(lion_to_audio(), True)
spectrogram2()
