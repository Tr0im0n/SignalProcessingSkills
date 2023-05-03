
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt
from scipy.io import wavfile
import sounddevice as sd
from scipy import optimize
# import os


def demodulate(signal, carrier_frequency, time_length, final_length):
    time_values = np.linspace(0, time_length, len(signal))
    signal *= np.cos(2 * np.pi * carrier_frequency * time_values)
    return resample(signal, final_length) - 0.5


def display(signal):
    plt.plot(signal)  # Plotting the signal keeps the audio playing in Python
    sd.play(signal / 25, audio_frequency)  # The division by 25 reduces the volume 25 times.
    plt.show()  # In Python, we need to "show" the plot after plotting.
    plt.close()


def moving_median_keep_ends(signal, window=2):
    # assert window % 2 == 1, "give odd number"
    smoothed = np.zeros_like(signal)
    len_signal = len(signal)
    # side = int((window-1)/2)
    for i in range(len_signal):
        if i <= window:
            smoothed[i] = np.median(signal[:i+window+1])
        elif len_signal - i < window:
            smoothed[i] = np.median(signal[i-window:])
        else:
            smoothed[i] = np.median(signal[i-window:i+window+1])
    return smoothed


def sine_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


def sine_func_ab(x, a, b):
    return a * np.sin(b * x)


def waves(n=4):
    xs = np.arange(output_size)
    # ys = np.zeros(output_size)
    y1 = np.sin(2*np.pi/output_size*xs)
    y2 = 0.5 * np.sin(4*np.pi/output_size*xs)
    y3 = 0.25 * np.sin(8 * np.pi / output_size * xs)
    ys = y1 * y2 * y3
    plt.plot(ys)
    plt.show()


# imported
def butter_lowpass_filter(data):
    nyq = 0.5 * audio_frequency
    cutoff = 1000
    normal_cutoff = cutoff / nyq
    order = 4
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='lowpass')
    y = filtfilt(b, a, data)
    return y


file_name = 'ghost_ship_transmission.wav'
fs, ghostly_signal = wavfile.read(file_name)

audio_frequency = 44_100  # this is 44.1 kHz
transmission_frequency = 4_410_000  # this is 4,410,000 Hz or 4,410 kHz
output_size = 105_000  # this is the size specified in problem 1
transmission_time = len(ghostly_signal) / transmission_frequency


def q1():
    carrier_frequency = 1_000_000
    output = demodulate(ghostly_signal, carrier_frequency, transmission_time, output_size)
    return output


def q2():
    carrier_frequency = 1_050_000
    output = demodulate(ghostly_signal, carrier_frequency, transmission_time, output_size)
    output = moving_median_keep_ends(output)
    volume = 15
    output = volume * output / max(output)
    return output


def q3():
    carrier_frequency = 1_100_000
    output = demodulate(ghostly_signal, carrier_frequency, transmission_time, output_size)
    # fit a sine wave
    data_points = 200
    x_data = np.arange(data_points)
    # params, cov = optimize.curve_fit(sine_func, x_data, output[:data_points], p0=[15, 0.02, 0, 0])
    params, cov = optimize.curve_fit(sine_func_ab, x_data, output[:data_points], p0=[15, 0.02])
    print(params)
    x_data_long = np.arange(output_size)
    sine_wave = sine_func_ab(x_data_long, *params)
    output = output - sine_wave
    volume = 15
    output = volume * output / max(output)
    return output


def q4():
    carrier_frequency = 1_150_000
    output = demodulate(ghostly_signal, carrier_frequency, transmission_time, output_size)

    # xs = np.arange(output_size)
    # y1 = 11 * np.sin(2 * np.pi / output_size * xs)
    # y2 = -y1
    # plt.plot(y1)
    # plt.plot(y2)
    # half_len = int(output_size/2)
    # plt.vlines(half_len, -10, 10)
    # ensemble = output[:half_len] + output[:half_len-1:-1] / 2
    # ensemble2 = np.append(ensemble, -ensemble[::-1])
    # plt.plot(ensemble2)
    # output = output - ensemble2

    # butter
    output = butter_lowpass_filter(output)

    # get rid of peak in beginning
    output = output[100:]

    # output = moving_median_keep_ends(output, 1)
    output = output / max(output)
    volume = 15
    output *= volume
    return output


def q5():
    carrier_frequency = 1_200_000
    output = demodulate(ghostly_signal, carrier_frequency, transmission_time, output_size)
    output = output[::-1]
    return output


def q6():
    carrier_frequency = 1_250_000
    output = demodulate(ghostly_signal, carrier_frequency, transmission_time, output_size)
    # on line 19 divide audio_frequency by 5
    return output


# 1. diluted rum
# 2. laser sharks
# 3. mutated seagulls
# 4. exploding cannon
# 5. parrot mutiny
# 6. dolphins with machetes
display(q3())
