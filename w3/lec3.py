
import numpy as np
from matplotlib import pyplot as plt


def square_wave(N=200):
    frequency = 1001
    xs = np.linspace(0, 1, frequency)
    y = np.zeros(frequency)
    for i in range(N):
        n = i*2 + 1
        y_new = np.sin(2*np.pi*xs*n)/n
        y = y + y_new
    plt.plot(xs, y)
    plt.show()


square_wave()



