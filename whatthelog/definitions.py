import os
from pathlib import Path


PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
PREFIX_TREE_PICKLE_PATH = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle")
CONFIG_FILE = PROJECT_ROOT.joinpath("resources/config.json")