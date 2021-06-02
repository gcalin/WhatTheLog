import os
from pathlib import Path
import random
import shutil
import sys

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT


sys.path.append("./../")

random.seed(4)


def main():
    DatasetFactory(PROJECT_ROOT.joinpath("resources/traces_large/")).create_data_set(10, 5, 5,
                                                                                     remove_trivial_loops=True,
                                                                                     visualize_tree=True,
                                                                                     name_tree="prefix_tree_original.png")
    print("Finished!")


if __name__ == "__main__":
    main()
