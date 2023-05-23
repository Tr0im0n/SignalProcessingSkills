import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

png_name = "data.png"
image_file = Image.open(png_name)
image_array = np.asarray(image_file)

csv_name = "spectra.csv"
csv_array = np.loadtxt(csv_name, delimiter=',', skiprows=0)


def show():
    plt.imshow(image_array, cmap="Greys_r")
    plt.show()

    for col in range(csv_array.shape[1]):
        column_data = csv_array[:, col]
        plt.plot(column_data, label=col)
    plt.legend()
    plt.show()


def anim():
    xs = np.arange(0, 640, 1)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 640), ylim=(0, 6.6e4))
    line, = ax.plot([], [])     # , lw=3

    def animate(frame):
        line.set_data(xs, image_array[:, frame])
        ax.set_title(f"Frame: {frame}")
        return line,

    animation = FuncAnimation(fig,
                              func=animate,
                              frames=640,
                              interval=30)
    plt.show()


def something():
    small_square_mat = np.matmul(csv_array.transpose(), csv_array)
    return np.matmul(np.matmul(np.linalg.inv(small_square_mat), csv_array.transpose()), image_array)


something_array = something()


def show2():
    for i, row in enumerate(something()):
        plt.plot(row, label=i)
    plt.legend()
    plt.show()


def gaussian(x, amp, m0, sigma, baseline=0):
    return amp*np.exp(pow(x-m0, 2)/(-1*pow(sigma, 2))) + baseline


def cos(x, amp, wavelength, baseline):
    return amp*np.cos(2*np.pi*x/wavelength) + baseline


def cos2(x, amp, wavelength):
    return amp*pow(np.cos(np.pi*x/wavelength), 2)


def gauss_cos2(x, amp, m0, sigma, wavelength):
    return gaussian(x, amp, m0, sigma, 0)*pow(np.cos(np.pi*x/wavelength), 2)


def gauss_lin(x, amp, m0, sigma):
    return x*amp*np.exp(pow(x-m0, 2)/(-1*pow(sigma, 2)))


def divide(x, amp, offset):
    return amp/(x+offset)


def exponent(x, amp, width):
    return amp*np.exp(-x/width)


def gauss_gauss(x, amp1, m01, sigma1, amp2, m02, sigma2):
    return gaussian(x, amp1, m01, sigma1) + gaussian(x, amp2, m02, sigma2)


def gauss_sin2(x, amp, m0, sigma, wavelength):
    return gaussian(x, amp, m0, sigma, 0)*pow(np.sin(np.pi*x/wavelength), 2)


# 0 ax+b
# 1 gauss
# 2 cos
# 3 ln or 1/x, e^-x
# 4 x^3
# 5 2 gaussians
# 6 cos * gauss

def fit0(show=False):
    ys = something_array[0]
    xs = np.arange(len(ys))
    p = np.polyfit(xs, ys, 1)
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, np.polyval(p, xs), label="fit")
        plt.title("0")
        plt.legend()
        plt.show()
    return p


def fit1(show=False):
    ys = something_array[1]
    xs = np.arange(len(ys))
    p_start = 30_000, 300, 100, 1
    lower = 1, 1, 1, 1
    upper = 100_000, 1000, 1000, 1000
    popt, _ = curve_fit(gaussian, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, gaussian(xs, *popt), label="fit")
        plt.title("1")
        plt.legend()
        plt.show()
    return popt


def fit2(show=False):
    ys = something_array[2]
    xs = np.arange(len(ys))
    p_start = 15_000, 100, 15_000
    lower = 1_000, 10, 1_000
    upper = 100_000, 1000, 100_000
    popt, _ = curve_fit(cos, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, cos(xs, *popt), label="fit")
        plt.title("2")
        plt.legend()
        plt.show()
    return popt


def fit22(show=False):
    ys = something_array[2]
    xs = np.arange(len(ys))
    p_start = 30_000, 100
    lower = 1_000, 10
    upper = 100_000, 1000
    popt, _ = curve_fit(cos2, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, cos2(xs, *popt), label="fit")
        plt.title("2")
        plt.legend()
        plt.show()
    return popt


def fit3divide(show=False):
    ys = something_array[3]
    xs = np.arange(len(ys))
    p_start = 300_000, 50
    lower = 100_000, 20
    upper = 10_000_000, 1000
    popt, _ = curve_fit(divide, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, divide(xs, *popt), label="fit")
        plt.title("3")
        plt.legend()
        plt.show()
    return popt


def fit3poly(show=False):
    ys = something_array[3]
    xs = np.arange(len(ys))
    p = np.polyfit(xs, ys, 3)
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, np.polyval(p, xs), label="fit")
        plt.title("3")
        plt.legend()
        plt.show()
    return p


def fit3(show=False):
    ys = something_array[3]
    xs = np.arange(len(ys))
    p_start = 30_000, 100
    lower = 10_000, 10
    upper = 100_000, 1_000
    popt, _ = curve_fit(exponent, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, exponent(xs, *popt), label="fit")
        plt.title("3")
        plt.legend()
        plt.show()
    return popt


def fit4(show=False):
    ys = something_array[4]
    xs = np.arange(len(ys))
    p = np.polyfit(xs, ys, 3)
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, np.polyval(p, xs), label="fit")
        plt.title("4")
        plt.legend()
        plt.show()
    return p


def fit5(show=False):
    ys = something_array[5]
    xs = np.arange(len(ys))
    p_start = 25_000, 400, 50, 12_000, 500, 50
    lower = 10_000, 10, 10, 5_000, 10, 10
    upper = 30_000, 1000, 100, 20_000, 1000, 100
    popt, _ = curve_fit(gauss_gauss, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, gauss_gauss(xs, *popt), label="fit")
        plt.title("5")
        plt.legend()
        plt.show()
    return popt


def fit52(show=False):
    ys = something_array[5]
    xs = np.arange(len(ys))
    p_start = 100, 400, 100
    lower = 1, 1, 1
    upper = 10_000, 1000, 500
    popt, _ = curve_fit(gauss_lin, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, gauss_lin(xs, *popt), label="fit")
        plt.title("5")
        plt.legend()
        plt.show()
    return popt


def fit6(show=False):
    ys = something_array[6]
    xs = np.arange(len(ys))
    p_start = 30_000, 320, 140, 64
    lower = 20_000, 100, 100, 50
    upper = 40_000, 500, 200, 80
    popt, _ = curve_fit(gauss_sin2, xs, ys, p0=p_start, bounds=(lower, upper))
    if show:
        plt.plot(xs, ys, label="data", lw=3)
        plt.plot(xs, gauss_sin2(xs, *popt), label="fit")
        plt.title("6")
        plt.legend()
        plt.show()
    return popt


def show_all_fits():
    funcs = fit0, fit1, fit2, fit22, fit3, fit3poly, fit4, fit5, fit6
    for func in funcs:
        func(True)


show2()
show_all_fits()
