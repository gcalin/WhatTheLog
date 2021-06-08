import os
from pathlib import Path
import random
import shutil
import sys

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer


sys.path.append("./../")

random.seed(1)


def main():

    # pt = PrefixTreeFactory().get_prefix_tree(PROJECT_ROOT.joinpath("resources/test/"), PROJECT_ROOT.joinpath("resources/config.json"))
    DatasetFactory(PROJECT_ROOT.joinpath("resources/traces_large/")).create_data_set(20, 5, 5,
                                                                                 remove_trivial_loops=False,
                                                                                 visualize_tree=True,
                                                                                 name_tree="prefix_tree_original.png")
    print("Finished!")


if __name__ == "__main__":
    main()
