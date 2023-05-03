
from scipy import signal
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class Sobel:
    LEFT = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    TOP = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])


def make_edge_array_greyscale(image_array: np.ndarray):
    left_sobel = signal.convolve(image_array, Sobel.LEFT)
    top_sobel = signal.convolve(image_array, Sobel.TOP)
    total_sobel = np.sqrt(pow(left_sobel, 2) + pow(top_sobel, 2))
    return total_sobel


def make_edge_array_rgb(image_array: np.ndarray):
    rgb = [image_array[:, :, i] for i in [0, 1, 2]]
    edges_list = [signal.convolve(color, sobel, mode="same") for color in rgb for sobel in [Sobel.LEFT, Sobel.TOP]]
    # print("start")
    # total_sobel = np.sum([pow(edge_array, 2)] for edge_array in edges_list)
    # print("stop")
    total_sobel = np.zeros_like(edges_list[0])
    for edge in edges_list:
        total_sobel += pow(edge, 2)
    return np.sqrt(total_sobel)


def make_edge_array(image_array: np.ndarray):
    if len(image_array[0, 0]) == 1:
        return make_edge_array_greyscale(image_array)
    elif len(image_array[0, 0]) == 3:
        return make_edge_array_rgb(image_array)
    else:
        print("something went wrong")
        return -1


def least_edgy_seam(edge_array: np.ndarray, return_arrays=False):
    # dynamic programming: find least e path
    least_e = np.zeros_like(edge_array)
    least_e[0] = edge_array[0]     # make the end the same
    directions = np.zeros_like(edge_array, dtype=np.int32)
    rows, cols = edge_array.shape
    for row in range(1, rows):
        for col in range(cols):
            # find min_e & direction for different cases
            if col == 0:
                min_e = min(least_e[row - 1, 0:2])
                direction = np.argmin(least_e[row - 1, 0:2])
            elif col == cols-1:
                min_e = min(least_e[row - 1, cols-2:cols])
                direction = np.argmin(least_e[row - 1, cols-2:cols]) - 1
            else:
                min_e = min(least_e[row-1, col-1:col+2])
                direction = np.argmin(least_e[row-1, col-1:col+2]) - 1
            # change array
            least_e[row, col] = edge_array[row, col] + min_e
            directions[row, col] = col + direction

    # make seam list
    seam_list = [np.argmin(least_e[-1])]
    for row in range(rows-1, 0, -1):
        next_direction = directions[row, seam_list[-1]]
        seam_list.append(next_direction)

    if not return_arrays:
        return seam_list

    # make seam array
    seam_array = np.zeros_like(edge_array)
    for row, i in zip(seam_array, seam_list[::-1]):
        row[i] = 1

    return seam_list, least_e, seam_array


def carve_seam(image_array: np.ndarray) -> np.ndarray:
    edge_array = make_edge_array(image_array)
    seam_list = least_edgy_seam(edge_array)
    rows, cols, rgb = np.shape(image_array)

    carved = np.zeros(shape=(rows, cols-1, rgb), dtype=np.int32)
    for row, col in enumerate(seam_list[:-1]):
        if col == 0:
            carved[row] = image_array[row, 1:, :]
        else:
            carved[row] = np.concatenate((image_array[row, :col, :], image_array[row, col+1:, :]))
    return carved

######################################################################################################################


def lion1():    # edges
    file_name = "lion.png"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    lion_left = signal.convolve(image_array, Sobel.LEFT)
    lion_top = signal.convolve(image_array, Sobel.TOP)
    lion_edges = np.sqrt(pow(lion_left, 2) + pow(lion_top, 2))

    fig, axs = plt.subplots(1, 4, sharex="all", sharey="all")
    for axis, image in zip(axs, [image_array, lion_left, lion_top, lion_edges]):     # .flatten()
        axis.imshow(image, cmap="Greys")
    plt.show()


def lion2():    # show first seam
    file_name = "lion.png"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    seam_list, least_e, seam_array = least_edgy_seam(image_array, True)

    fig, axs = plt.subplots(1, 3, sharex="all", sharey="all")
    for axis, image in zip(axs, [image_array, least_e, seam_array]):  # .flatten()
        axis.imshow(image, cmap="Greys")
    plt.show()


def lion3():
    file_name = "lion.png"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    carved = carve_seam(image_array)
    for _ in range(9):
        carved = carve_seam(carved)

    print(np.shape(image_array), np.shape(carved))

    fig, axs = plt.subplots(1, 2, sharex="all", sharey="all")
    for axis, image in zip(axs, [image_array, carved]):  # .flatten()
        axis.imshow(image, cmap="Greys")
    plt.show()


def dali1():    # show colors
    file_name = "dali.bmp"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    edge_array = make_edge_array_rgb(image_array)
    plt.imshow(edge_array, cmap="Greys")
    plt.show()

    fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")
    for axis, image, color in zip(axs.flatten(), [image_array, red, blue, green],
                                  ["Greys", "Reds_r", "Greens_r", "Blues_r"]):   # .flatten()
        axis.imshow(image, cmap=color)
        axis.set_title(color)
    plt.show()


def dali2(n=10):
    file_name = "dali.bmp"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)

    carved = np.copy(image_array)
    for _ in range(n):
        carved = carve_seam(carved)

    fig, axs = plt.subplots(1, 2, sharey="all")
    for axis, image, title in zip(axs, [image_array, carved], ("Original", f"Seam carved {n} times")):
        axis.set_title(title)
        axis.imshow(image)
    plt.show()


dali1()
