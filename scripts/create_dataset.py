import os
from pathlib import Path
import random
import shutil
import sys

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT


sys.path.append("./../")

random.seed(1)


def main():
    DatasetFactory(PROJECT_ROOT.joinpath("resources/traces_5/")).create_data_set(5, 5, 5,
                                                                                     remove_trivial_loops=False,
                                                                                     visualize_tree=True)
    print("Finished!")


if __name__ == "__main__":
    main()
