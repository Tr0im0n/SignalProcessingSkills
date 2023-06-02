
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd


# edge detection --------------------------------------------------------------

class Sobel:
    LEFT = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])
    TOP = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])


def make_edge_array(image_array: np.ndarray) -> np.ndarray:
    if len(np.shape(image_array)) == 2:
        edges_list = [signal.convolve(image_array, sobel, mode="same")
                      for sobel in [Sobel.LEFT, Sobel.TOP]]
    else:
        edges_list = [signal.convolve(image_array[:, :, color], sobel, mode="same")
                      for color in range(image_array.shape[2]) for sobel in [Sobel.LEFT, Sobel.TOP]]

    return np.sqrt(sum([pow(edge, 2)for edge in edges_list]))


def make_edge_array2(image_array: np.ndarray) -> np.ndarray:
    if len(np.shape(image_array)) == 2:
        edges_list = []
        for sobel in [Sobel.LEFT, Sobel.TOP]:
            edges_list.append(signal.convolve(image_array, sobel, mode="same"))
    else:
        edges_list = []
        for color in range(image_array.shape[2]):
            for sobel in [Sobel.LEFT, Sobel.TOP]:
                edges_list.append(signal.convolve(image_array[:, :, color], sobel, mode="same"))

    return np.sqrt(sum([pow(edge, 2)for edge in edges_list]))


def apple_core_example():
    os.chdir(r"../images")
    # file_name = "apple-core.jpg"
    # file_name = "AppleCore.jpg"
    file_name = "APPLE-CORE-BITTEN.jpg"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    edge_array = make_edge_array(image_array)
    my_vmax = 127
    plt.imshow(edge_array, vmax=my_vmax, aspect="equal")  # , cmap="Greys_r", vmax=255
    plt.title(f"RGB edge detection of apple core, vmax = {my_vmax}")
    plt.show()


# apple_core_example()

# matching noise --------------------------------------------------------------

def get_residual2(array1, array2):
    sr = 44100
    sample_len = int(1.95 * sr)
    ans = 0
    for i, j in zip(array1[:sample_len], array2[:sample_len]):
        ans += pow(i-j, 2)
    return ans


def find_best_bird():
    forest_bird = bird_in_forest_ensemble_average()
    parrot_lib_path = r"C:\Users\Tom\Documents\Universiteit\Signal Processing and Control Skills\week 2\parrot_library"
    os.chdir(parrot_lib_path)
    names = all_bird_file_names()
    min = 1e5
    min_index = -1
    for index, name in enumerate(names):
        _, data = wavfile.read(name)
        max_amp = max(data)
        data1 = data/max_amp
        residual = get_residual(forest_bird, data1)
        if residual < min:
            min = residual
            min_index = index
    return min_index, min


def get_names() -> tuple:
    names = ("HDPE bottle recording .wav",
             "PET bottle recording .wav",
             "white_HDPE_bottle.wav",
             "unknown_bottle.wav")
    return names


def get_fourier_transforms() -> list:
    os.chdir(r"..\images")
    names = get_names()
    datas = []
    for name in names:
        _, data = wavfile.read(name)
        norm_data = data[:, 0] / max(data[:, 0])
        rfft_data = abs(np.fft.rfft(norm_data))
        shortened_rfft_data = rfft_data[:20_000]
        norm_rfft = shortened_rfft_data / max(shortened_rfft_data)
        datas.append(norm_rfft)
    return datas


def plot_drop_sound() -> None:
    datas = get_fourier_transforms()
    names = get_names()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle("Fourier transforms of 4 bottle dropping sounds")
    axs = [ax1, ax2, ax3, ax4]
    for ax, data, name in zip(axs, datas, names):
        ax.plot(data, label=f"maximum frequency = {np.argmax(data)} hz")
        ax.legend()
        ax.set_title(name)

    plt.legend()
    plt.show()


def get_residual(array1, array2):
    if not len(array1) == len(array2):
        print("Not same length")
    return sum([pow(i-j, 2) for i, j in zip(array1, array2)])


def get_all_residuals():
    datas = get_fourier_transforms()
    for i in [0, 1, 2]:
        res = get_residual(datas[i], datas[3])
        print(res)


def get_squared_residuals():
    datas = get_fourier_transforms()
    datas = [pow(data, 2) for data in datas]
    for i in [0, 1, 2]:
        res = get_residual(datas[i], datas[3])
        print(res)


def main():
    # apple_core_example()
    plot_drop_sound()
    get_all_residuals()
    get_squared_residuals()


if __name__ == "__main__":
    main()
