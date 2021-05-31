#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import timedelta
import os
from pathlib import Path
from sys import setrecursionlimit
from time import time
import numpy as np
import matplotlib.pyplot as plt

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from clustering.evaluator import Evaluator
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.clustering.state_model_factory import StateModelFactory
from whatthelog.auto_printer import AutoPrinter

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def plot(data, numXTicks, title, xLabel, yLabel):
    """
    Plot a single list of data points against their index.
    :param data: the list of data to plotNoX
    :param numXTicks: the number of ticks to show in the x axis
    :param title: the graph title
    :param xLabel: the X axis label
    :param yLabel: the Y axis label
    """

    xValues = np.arange(0, numXTicks, (numXTicks / len(data)))
    plt.plot(xValues, data)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def multiPlot(datasets, labels, numXTicks, title, xLabel, yLabel):
    """
    Plot multiple lists of data points against each points' index.
    :param datasets: the list of lists to plotNoX
    :param labels: the label for each data list
    :param numXTicks: the number of ticks to show in the x axis
    :param title: the graph title
    :param xLabel: the X axis label
    :param yLabel: the Y axis label
    """

    assert len(datasets) == len(labels), "Number of supplied labels does not match provided datasets"
    fig, ax = plt.subplots()
    ax.set(title=title,
           xlabel=xLabel,
           ylabel=yLabel)

    # Plot each line
    for x in range(len(datasets)):
        xValues = np.arange(0, numXTicks, (numXTicks / len(datasets[x])))
        ax.plot(xValues,
                datasets[x],
                label=labels[x],
                alpha=0.7,
                linewidth=2)

    plt.xticks(np.arange(start=0, stop=(numXTicks+1), step=5))
    ax.legend()
    plt.show()

if __name__ == '__main__':

    setrecursionlimit(int((10**6)/2))

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent
    start_time = time()

    pt = PrefixTreeFactory.unpickle_tree(project_root.joinpath('out/testPrefixTree.p'))

    factory = StateModelFactory(pt)
    model, data = factory.run_clustering('louvain')

    print(f"Done! Final model size: {len(model)}, time elapsed: {timedelta(seconds=time() - start_time)}")

    multiPlot(list(zip(*data)), ['specificity', 'recall', 'size'], len(data),
              'Metrics over merges', 'merge', 'metric')

    factory.pickle_model(model, project_root.joinpath('out/testModel.p'))
