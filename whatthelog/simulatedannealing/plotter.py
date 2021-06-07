import matplotlib.pyplot as plt
from tkinter import *
if __name__ == "__main__":
    filename = "SA_output.txt"
    sz, sp, re = [], [], []
    f = open(filename)
    for line in f.readlines():
        size, specificty, recall = line.split()
        sz.append(size)
        sp.append(specificty)
        re.append(recall)

    c = range(len(sz))

    plt.subplot(1, 3, 1)
    plt.plot(c, sz)

    plt.subplot(1, 3, 2)
    plt.plot(c, sp)

    plt.subplot(1, 3, 3)
    plt.plot(c, re)

    plt.show()
    f.close()