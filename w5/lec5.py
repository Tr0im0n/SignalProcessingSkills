
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, amp, sigma, m0):
    return amp*np.exp(pow(x-m0, 2)/(-1*pow(sigma, 2)))


def fit1(show=True):
    xs = np.linspace(0, 100, 1001)
    ys_orig = gaussian(xs, 3, 15, 50)
    ys = ys_orig + np.random.rand(len(ys_orig)) - 0.5

    p_start = 10, 10, 10
    lower = 1, 1, 1
    upper = 100, 100, 100
    popt, pcov = curve_fit(gaussian, xs, ys, p0=p_start, bounds=(lower, upper))

    if not show:
        return

    print(popt, pcov, sep="\n")
    plt.plot(xs, ys)
    plt.plot(xs, ys_orig)
    plt.plot(xs, gaussian(xs, *popt))
    plt.show()
    return popt, pcov


fit1()
