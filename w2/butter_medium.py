import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter,filtfilt

# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples


# Step 2 : Create some sample data with noise

# sin wave
t = np.linspace(0, T, n)
sig = np.sin(1.2*2*np.pi*t)
# Lets add some noise
noise = 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
data = sig + noise

# Step 3 : Filter implementation using scipy


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# Step 4 : Filter and plot the data

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)
plt.plot(data)
plt.plot(y)
plt.show()
exit()

fig = plt.figure()
plt.scatter(
            x = t,
            y = data,
            line =  dict(shape =  'spline' ),
            name = 'signal with noise'
            )
plt.scatter(
            y = y,
            line =  dict(shape =  'spline' ),
            name = 'filtered signal'
            )
fig.show()