### This is a code snippet to help with completing assignment 2.
### The intended output of this script is to plot and play
### the audio to answer question 1 of assignment 2.
### This script assumes it is in the same directory as the file
### "ghost_ship_transmission.wav" and that the libraries below
### are all installed. This is a Python script, but very
### similar code is possible with MATLAB or other scripting
### languages.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.io import wavfile
import sounddevice as sd
import os


def demodulate(signal, carrier_frequency, time_length, final_length):
    '''An updated demodulation function for assignment 2'''
    time_values = np.linspace(0, time_length, len(signal))
    signal *= np.cos(2 * np.pi * carrier_frequency * time_values)
    return resample(signal, final_length) - 0.5

# Below we are simply loading the "ghost_ship_transmission.wav" file
# MATLAB has a similar function called "audioread" used like:
# [ghostly_signal, fs] = audioread(wav_file)
# where "wav_file" is the file name of the wave file
directory = os.path.dirname(os.path.abspath(__file__))
wav_file = os.path.join(directory, 'ghost_ship_transmission.wav')
fs, ghostly_signal = wavfile.read(wav_file)

# We do some basic calculation
audio_frequency = 44_100 # this is 44.1 kHz
transmission_frequency = 4_410_000 # this is 4,410,000 Hz or 4,410 kHz
output_size = 105_000 # this is the size specified in problem 1
transmission_time = len(ghostly_signal) / transmission_frequency

# Below, we put in "1_000_000" as that is the frequency to demodulate at for
# question 1, it will change for questions 2 through 6. For example, it will
# be 1_050_000 for question 2.
output = demodulate(ghostly_signal, 1_000_000, transmission_time, output_size)

# for problems 2 through 6, you will need to "clean" or apply signal processing
# algorithms in this area of the code to enhance the audio of the "output" variable

plt.plot(output) # Plotting the signal keeps the audio playing in Python
sd.play(output / 25) # The division by 25 reduces the volume 25 times.
plt.show()       # In Python, we need to "show" the plot after plotting.
