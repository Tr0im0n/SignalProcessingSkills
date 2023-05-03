
import os
import csv
import struct
import numpy as np
from matplotlib import pyplot as plt

# OS stuff
# cur_dir = os.getcwd()
os.chdir(r"C:\Python\PycharmProjects\SignalProcessingSkills\assignment1data")

# Question 1 ########################################################################################################
print("Question 1")

# 0. reading the data
data_list = [[[], []], [[], []], [[], []], [[], []]]
with open('quartet.csv', newline='') as csvfile:
    spam_reader = csv.reader(csvfile, delimiter=',')
    for row in spam_reader:
        data_list[int(row[0])-1][0].append(float(row[1]))
        data_list[int(row[0])-1][1].append(float(row[2]))
# print(f"data_list: {data_list}")

# 1. mean of x & y cords of the 4 crew members
mean_list = [ [np.mean(crew_mate[0]), np.mean(crew_mate[1])] for crew_mate in data_list ]
print(f"mean_list: {mean_list}")

# 2. std of x and y of the 4 crew members
std_list = [ [np.std(crew_mate[0]), np.std(crew_mate[1])] for crew_mate in data_list ]
print(f"std_list: {std_list}")

# 3. linfit through x and y data
# fit_list = [ np.polyfit(crew_mate[0], crew_mate[1], 1) for crew_mate in data_list ]
fit_list = []
res_list = []
for crew_mate in data_list:
    coef, residuals, *_ = np.polyfit(crew_mate[0], crew_mate[1], 1, full=True)
    fit_list.append(coef)
    res_list.append(*residuals)
print(f"fit_list: {fit_list}")

# 4. correlation between x & y, Pearson, R2
print(f"res_list: {res_list}")

# 5. hypothesis of which crew mate is closest to the average

# 6. Plot and more? plot fit lines
def plot1():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="all", sharey="all")
    fig.suptitle("Figure of coordinates and the linear fit line")
    x = np.arange(0, 20, 0.1)

    ax1.plot(x, np.polyval(fit_list[0], x))
    ax1.scatter(data_list[0][0], data_list[0][1])
    ax1.set(ylabel="y")
    ax1.set_title("crew mate 1")
    ax2.plot(x, np.polyval(fit_list[1], x), 'tab:orange')
    ax2.scatter(data_list[1][0], data_list[1][1])
    ax2.set_title("crew mate 2")
    ax3.plot(x, np.polyval(fit_list[2], x), 'tab:green')
    ax3.scatter(data_list[2][0], data_list[2][1])
    ax3.set(xlabel="x", ylabel="y")
    ax3.set_title("crew mate 3")
    ax4.plot(x, np.polyval(fit_list[3], x), 'tab:red')
    ax4.scatter(data_list[3][0], data_list[3][1])
    ax4.set(xlabel="x")
    ax4.set_title("crew mate 4")

    plt.show()

plot1()

# Question 2 ########################################################################################################
print("\nQuestion 2")

with open("treasure.bin", mode='rb') as file:
    fileContent = file.read()
    # 16 1-byte int
    a = struct.unpack("<" + "b"*16, fileContent)
    # 16 1-byte hexadecimal
    b = [hex(i).lstrip("0x") for i in a]
    # Two 8-byte signed integers
    c = struct.unpack("<qq", fileContent)
    # Four 4-byte unsigned integers
    d = struct.unpack("<LLLL", fileContent)
    # Two 8-byte floating-point numbers
    e = struct.unpack("<dd", fileContent)

    print(a, b, c, d, e, sep="\n")

    # chr
    f = [chr(i) for i in a]
    print("".join(f))

    # binary
    bi = []
    for i in a:
        byte = f"{i+128:b}"
        byte = "0" + byte[1:]
        bi.append(byte)
        # print(byte, end=" & ")
    print(bi)

    a2 = "int"
    for i in a:
        a2 += f" & {i}"
    print(a2)

    b2 = "hex"
    for i in b:
        b2 += f" & {i}"
    print(b2)

    f2 = "chr"
    for i in f:
        f2 += f" & {i}"
    print(f2)

# Question 3 ########################################################################################################
print("\nQuestion 3")

with open("map.bin", mode='rb') as file:
    fileContent = file.read()
    a = struct.unpack("<"+"B"*88, fileContent)
    print("All coords: ", *a)
    print(f"Sum(x) = {sum(a[::2])}")
    print(f"Sum(y) = {sum(a[1::2])}")
    plt.plot(a[::2], a[1::2])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("NOT HERE")
    plt.show()
