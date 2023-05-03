import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import savgol_filter
from scipy.io import wavfile
import sounddevice as sd
import os

def moving_average_keep_ends(signal, window=3):
    assert window % 2 != 0, "Give an odd number for the window"
    smoothed = np.zeros_like(signal)
    for i, point in enumerate(signal):
        if i < window // 2 + 1:
            smoothed[i] = np.average(signal[: i + window // 2 + 1])
        elif len(smoothed) - i < window // 2 + 1:
            smoothed[i] = np.average(signal[i - window // 2 - 1:])
        else:
            smoothed[i] = np.average(signal[i - window // 2 - 1:i + window // 2 + 1 ])
    return smoothed

def moving_median_keep_ends(signal, window=3):
    assert window % 2 != 0, "Give an odd number for the window"
    smoothed = np.zeros_like(signal)
    for i, point in enumerate(signal):
        if i < window // 2 + 1:
            smoothed[i] = np.median(signal[: i + window // 2 + 1])
        elif len(smoothed) - i < window // 2 + 1:
            smoothed[i] = np.median(signal[i - window // 2 - 1:])
        else:
            smoothed[i] = np.median(signal[i - window // 2 - 1:i + window // 2 + 1 ])
    return smoothed

def moving_average_truncate_ends(signal, oddNumber):
    distance = int((oddNumber - 1) / 2)
    croppedSignal = signal[distance:-distance]   # End of signal is chopped off
    smoothedSignal = np.zeros_like(croppedSignal)
    for i in range(len(croppedSignal)):
        smoothedSignal[i] = np.average(signal[i:i + oddNumber])
    return smoothedSignal

def add_random_noise(signal):
    return (signal + (np.random.random(signal.size) - 0.5)) / 2

def add_shot_noise(signal):
    for i in range(len(signal)):
        add_noise = np.random.random(1)
        if add_noise > 0.999:
            signal[i] += (np.random.random(1) * 5) - 2.5
    return signal

def normalize(signal):
    return signal / np.max(signal)

directory = os.path.dirname(os.path.abspath(__file__))
fs, data1 = wavfile.read(os.path.join(directory, 'beat_pattern.wav'))
fs, data2 = wavfile.read(os.path.join(directory, 'white_noise.wav'))

data1 = normalize(data1)
x = np.linspace(0, 3, fs * 3)

shot = add_shot_noise(data1)
filtered = moving_median_keep_ends(shot)
plt.plot(data1)
plt.plot(filtered)
sd.play(filtered)
plt.show()
