import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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


anim()
