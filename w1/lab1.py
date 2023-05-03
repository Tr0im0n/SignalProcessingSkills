
import csv
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

os.chdir(r"C:\Matlab\scripts")

data_table = []
with open('mat1.csv', newline='') as csvfile:
    spam_reader = csv.reader(csvfile, delimiter=',')
    for row in spam_reader:
        new_row = [ float(i) for i in row]
        data_table.append(new_row)

mag_list = [ np.sqrt( pow(row[0], 2) + pow(row[1], 2) + pow(row[2], 2) ) - 9.81 for row in data_table ]
list_len = len(mag_list)

smooth_amount = 5
smooth_range = list_len//smooth_amount
smooth1 = [ np.mean(mag_list[i*smooth_amount:(i+1)*smooth_amount]) for i in range(smooth_range) ]

central_deriv = [ mag_list[i] - mag_list[i-1] for i in range(list_len) ]
smooth_deriv = [ (smooth1[i+1]-smooth1[i-1])/2 for i in range(smooth_range-1) ]
# i = 0 --> i-1 = -1


def peak_finder(signal, deriv, amplitude_thresh=2, slope_thresh=2):
    peaks = []
    for index in range(len(deriv)):
        if signal[index] > deriv[index] and signal[index] >= amplitude_thresh:  # deriv[index-1] -
            peaks.append(index)
    return peaks


def peak_finder2(signal, amplitude_thresh=2):
    ans = []
    for i in range(len(signal)-1):
        if signal[i]>signal[i-1] and signal[i]>signal[i+1] and signal[i] >= amplitude_thresh:
            ans.append(i)
    return ans


peak_list = peak_finder2(smooth1)   # , smooth_deriv
print(peak_list)
print(len(peak_list))

plt.hlines([0], 0, 120, colors='k')
plt.vlines(peak_list, -5, 5, colors='k')
# plt.plot(mag_list, label="mag_list")
# plt.plot(central_deriv, label="central_deriv")
plt.plot(smooth1, label="smooth1")
# plt.plot(smooth_deriv, label="smooth_deriv")
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()
plt.close()



def anim_func(save=False):
    fig = plt.figure()

    n = 100  # Number of frames
    x = ['x', 'y', 'z']
    barcollection = plt.bar(x, data_table[1])

    def animate(t):
        y = data_table[t+1]
        for index, b in enumerate(barcollection):
            b.set_height(y[index])
        return barcollection

    anim = FuncAnimation(fig, animate, repeat=False, blit=True, frames=n, interval=100)
    plt.show()

# anim_func()
