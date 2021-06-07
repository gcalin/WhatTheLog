from typing import List


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filename = "output_kfold.txt"
    sz: List[float] = []
    sp: List[float] = []
    re: List[float] = []
    f = open(filename)
    for line in f.readlines():
        _, size, specificty, recall = line.split()
        sz.append(1-float(size))
        sp.append(1-float(specificty))
        re.append(1-float(recall))
    f.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    c = range(len(sp))

    ax.scatter3D(sp, re, sz, c=sz, marker="^", cmap="Greens")
    plt.xlabel("size")
    plt.ylabel("specificity")
    ax.set_zlabel("recall")

    filename = "output_kfold_500.txt"
    sz = []
    sp = []
    re = []
    f = open(filename)
    for line in f.readlines():
        _, size, specificty, recall = line.split()
        sz.append(1 - float(size))
        sp.append(1 - float(specificty))
        re.append(1 - float(recall))
    f.close()
    ax.scatter3D(sp, re, sz, c=sz, marker="o", cmap="Reds")
    plt.show()