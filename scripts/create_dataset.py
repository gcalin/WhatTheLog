import os
from pathlib import Path
import random
import shutil
import sys

sys.path.append("./../")

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
import scripts.log_scrambler

random.seed(5)

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
ALL_TRACES_PATH = PROJECT_ROOT.joinpath("resources/traces_large")
DATA_PATH = PROJECT_ROOT.joinpath("resources/data/")
CONFIG_FILE_PATH = PROJECT_ROOT.joinpath("resources/config.json")
PREFIX_TREE_PICKLE_PATH = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle")

TRACES_AMOUNT = 1000
POSITIVE_TRACES_AMOUNT = 100
NEGATIVE_TRACES_AMOUNT = 100


def main():
    # Get all logs
    logs = os.listdir(ALL_TRACES_PATH)

    # Clear data directory
    for file in os.listdir(DATA_PATH):
        shutil.rmtree(DATA_PATH.joinpath(file))

    # Create folders
    os.mkdir(DATA_PATH.joinpath("traces"))
    os.mkdir(DATA_PATH.joinpath("positive_traces"))
    os.mkdir(DATA_PATH.joinpath("negative_traces"))

    # Create traces
    for i in range(TRACES_AMOUNT):
        rand = random.randint(0, len(logs) - 1)
        shutil.copy(ALL_TRACES_PATH.joinpath(logs[rand]), DATA_PATH.joinpath(f"traces/xx{i}"))
        logs.remove(logs[rand])

    # Create positive traces
    for i in range(POSITIVE_TRACES_AMOUNT):
        rand = random.randint(0, len(logs) - 1)
        shutil.copy(ALL_TRACES_PATH.joinpath(logs[rand]), DATA_PATH.joinpath(f"positive_traces/positive_xx{i}"))
        logs.remove(logs[rand])

    # pt = PrefixTreeFactory.get_prefix_tree(DATA_PATH.joinpath("traces"), CONFIG_FILE_PATH)

    # PrefixTreeFactory.pickle_tree(pt, PREFIX_TREE_PICKLE_PATH)
    pt = PrefixTreeFactory.unpickle_tree(PREFIX_TREE_PICKLE_PATH)
    st = SyntaxTreeFactory().parse_file(CONFIG_FILE_PATH)

    traces = os.listdir(DATA_PATH.joinpath("traces"))

    for i in range(NEGATIVE_TRACES_AMOUNT):
        rand = random.randint(0, len(traces) - 1)
        input_file = DATA_PATH.joinpath("traces/" + traces[rand])
        output_file = DATA_PATH.joinpath(f"negative_traces/negative_xx{i}")

        scripts.log_scrambler.produce_false_trace(input_file, output_file, st, pt)

        traces.remove(traces[rand])


if __name__ == "__main__":
    main()
