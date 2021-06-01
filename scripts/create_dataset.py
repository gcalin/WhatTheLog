import os
from pathlib import Path
import random
import shutil
import sys

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT


sys.path.append("./../")

random.seed(5)


def main():
    DatasetFactory(PROJECT_ROOT.joinpath("resources/traces_large/")).create_data_set(10, 5, 5)
    print("Finished!")


if __name__ == "__main__":
    main()
