import struct

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


file_name = "lion.png"
image_file = Image.open(file_name)
image_array = np.asarray(image_file)


def test1():
    outside = []
    for row in image_array:
        for ele in row:
            if ele > 255 or ele < 0:
                outside.append(ele)
    print(outside)


def test2():
    for row in image_array:
        plt.plot(row)
    plt.show()


def test3():
    byte_array = array0.tobytes()
    # print(byte_array[460:])
    # print(array0[460:])
    byte_array2 = struct.pack('B'*534, *array0)
    print(byte_array2)
    print(struct.pack('b', 126))


def test4():
    for i in range(256):
        print(f"{i} & {struct.pack('B', i)}")


def test5():
    with open("filename.bin", "wb") as new_file:
        # new_file_byte_array = bytes(image_array[0])
        array0 = image_array[0]
        byte_array2 = struct.pack('B' * 534, *array0)
        print(byte_array2)
        new_file.write(byte_array2)

    with open("filename.bin", "rb") as old_file:
        file_bytes = old_file.read()
        a = struct.unpack("B" * 534, file_bytes)
        print(a)


def test6():
    with open("filename.bin", "wb") as new_file:
        for row in image_array:
            new_file.write(struct.pack('B'*534, *row))

    with open("filename.bin", "rb") as old_file:
        new_array = np.fromfile(old_file, np.dtype('u1'))
        print(new_array[:10])
        new2_array = np.reshape(new_array, (680, 534))
        plt.imshow(new2_array, cmap="Greys_r")
        plt.show()


def test7():
    dali_file_name = "dali.bmp"
    dali_file = Image.open(dali_file_name)
    dali_array = np.asarray(dali_file)
    shape = dali_array.shape
    print(shape)

    with open("filename2.bin", "wb") as new_file:
        for row in dali_array:
            for color in np.transpose(row):
                new_file.write(struct.pack('B'*shape[1], *color))

    with open("filename2.bin", "rb") as old_file:
        new_array = np.fromfile(old_file, np.dtype('u1'))
        print(new_array[:10])
        new2_array = np.reshape(new_array, (570, 3, 750))
        new3_array = [np.transpose(row) for row in new2_array]
        plt.imshow(new3_array)
        plt.show()


test2()
