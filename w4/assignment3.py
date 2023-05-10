import struct

import numpy as np
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt


# Tools #############################################################################

class Sobel:
    LEFT = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    TOP = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])


def make_edge_array(image_array: np.ndarray):
    if len(np.shape(image_array)) == 2:
        edges_list = [signal.convolve(image_array, sobel, mode="same")
                      for sobel in [Sobel.LEFT, Sobel.TOP]]
    else:
        edges_list = [signal.convolve(image_array[:, : color], sobel, mode="same")
                      for color in np.shape[2] for sobel in [Sobel.LEFT, Sobel.TOP]]

    return np.sqrt(sum([pow(edge, 2)for edge in edges_list]))


# Questions ##############################################################

def lion1():    # edges
    file_name = "lion.png"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    lion_left = signal.convolve(image_array, Sobel.LEFT)
    lion_top = signal.convolve(image_array, Sobel.TOP)
    lion_edges = np.sqrt(pow(lion_left, 2) + pow(lion_top, 2))

    fig, axs = plt.subplots(1, 4, sharex="all", sharey="all")
    for axis, image in zip(axs, [image_array, lion_left, lion_top, lion_edges]):     # .flatten()
        axis.imshow(image, cmap="Greys_r")
    plt.show()


def compress1():
    with open("Dali.bmp", mode='rb') as file:
        data = bytearray(file.read())
        offset = 0
        a = struct.unpack_from("b"*64, data)
        print(a)
        for line in data:
            print(len(line))


compress1()
