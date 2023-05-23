
import struct
import numpy as np
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt
import cv2


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


def subsample2d(array):
    old_shape = array.shape
    new_shape = int(old_shape[0]/2), int(old_shape[1]/2)
    subsample = np.zeros(new_shape, dtype=np.dtype('B'))
    for row in np.arange(new_shape[0]):
        for col in np.arange(new_shape[1]):
            sumi = sum([array[2*row+1, 2*col+1], array[2*row, 2*col+1], array[2*row+1, 2*col], array[2*row, 2*col]])
            subsample[row, col] = int(sumi/4)
    return subsample


def upsample2d(array):
    old_shape = array.shape
    new_shape = old_shape[0]*2, old_shape[1]*2
    upsampled = np.zeros(new_shape, dtype=np.dtype('B'))
    for row in np.arange(0, new_shape[0]):
        for col in np.arange(1, new_shape[1]):
            upsampled[row, col] = array[int(row/2), int(col/2)]
    return upsampled


def chroma_subsample(rgb_img):
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCR_CB)
    y = ycrcb_img[:, :, 0]
    cr = subsample2d(ycrcb_img[:, :, 1])
    cb = subsample2d(ycrcb_img[:, :, 2])
    return y, cr, cb


def combine_ycrcb(y, cr, cb):
    shape_y = y.shape
    ans = np.zeros((*shape_y, 3), dtype=np.dtype('B'))
    ans[:, :, 0] = y
    ans[:, :, 1] = cr
    ans[:, :, 2] = cb
    return ans


def rle_row(row):
    ans = [row[0]]
    counter = 0
    for num in row:
        if num != ans[-1]:
            ans.append(counter)
            counter = 1
            ans.append(num)
        else:
            counter += 1
    ans.append(counter)
    return ans


def rld(string):
    ans = []
    for num, counter in zip(string[0::2], string[1::2]):
        ans += [num]*counter
    return ans


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


def test1():
    file_name = "dali.bmp"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)
    y, cr, cb = chroma_subsample(image_array)

    with open("filename4.bin", "wb") as new_file:
        for color in y, cr, cb:
            for row in color:
                new_file.write(struct.pack('B'*len(row), *row))

    with open("filename4.bin", "rb") as old_file:
        new_array = np.fromfile(old_file, np.dtype('u1'))

    print(new_array.shape)
    y = new_array[:427_500]
    y = np.reshape(y, (570, 750))
    cr_sub = new_array[427_500:534_375]
    cr_sub = np.reshape(cr_sub, (285, 375))
    cb_sub = new_array[534_375:]    # 641_250
    cb_sub = np.reshape(cb_sub, (285, 375))
    cr = upsample2d(cr_sub)
    cb = upsample2d(cb_sub)
    ycrcb = combine_ycrcb(y, cr, cb)

    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)

    plt.imshow(rgb)
    plt.show()


def test2():    # with Run length encoding
    file_name = "dali.bmp"
    image_file = Image.open(file_name)
    image_array = np.asarray(image_file)
    y, cr, cb = chroma_subsample(image_array)

    with open("filename4.bin", "wb") as new_file:
        for color in y, cr, cb:
            for row in color:
                rled_row = rle_row(row)
                new_file.write(struct.pack('B'*len(rled_row), *rled_row))

    with open("filename4.bin", "rb") as old_file:
        new_array = np.fromfile(old_file, np.dtype('u1'))

    new_array2 = rld(new_array)
    new2_array = np.reshape(new_array2, (855, 750))

    y = new2_array[:570]
    crcb = new2_array[570:].reshape(570, 375)
    cr_sub = crcb[:285]
    cb_sub = crcb[285:]
    cr = upsample2d(cr_sub)
    cb = upsample2d(cb_sub)
    ycrcb = combine_ycrcb(y, cr, cb)

    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)

    plt.imshow(rgb)
    plt.show()


test1()
