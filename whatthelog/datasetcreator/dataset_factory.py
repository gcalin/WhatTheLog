import os
import random
import shutil
import sys
from pathlib import Path
from typing import Tuple, List

from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


sys.path.append("./../")

from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.datasetcreator.log_scrambler import LogScrambler


random.seed(5)


class DatasetFactory:
    DATA_PATH = PROJECT_ROOT.joinpath("resources/data/")

    def __init__(self, all_traces_path: str,
                 config_file_path: str = PROJECT_ROOT.joinpath(
                     "resources/config.json"),
                 prefix_tree_pickle_path: str = PROJECT_ROOT.joinpath(
                     "resources/prefix_tree.pickle")) -> None:
        self.all_traces_path = all_traces_path
        self.config_file_path = config_file_path
        self.prefix_tree_pickle_path = prefix_tree_pickle_path

    def create_data_set(self, traces_amount: int, positive_traces_amount: int,
                        negative_traces_amount: int, unique_graph, remove_trivial_loops: bool = True,
                        visualize_tree: bool = False,
                        name_tree: str = "prefix_tree_original.png") -> None:
        logs = os.listdir(self.all_traces_path)
        logs = [Path(self.all_traces_path).joinpath(line) for line in logs]

        # Clear data directory
        for file in os.listdir(self.DATA_PATH):
            shutil.rmtree(self.DATA_PATH.joinpath(file))

        # Create folders
        os.mkdir(self.DATA_PATH.joinpath("traces"))
        os.mkdir(self.DATA_PATH.joinpath("positive_traces"))
        os.mkdir(self.DATA_PATH.joinpath("negative_traces"))

        syntax_tree = SyntaxTreeFactory().parse_file(self.config_file_path)

        training_set = []
        # Create traces
        for i in range(traces_amount):
            rand = random.randint(0, len(logs) - 1)
            shutil.copy(logs[rand],
                        self.DATA_PATH.joinpath(f"traces/xx{i}"))
            training_set.append(logs[rand])
            logs.remove(logs[rand])

        prefix_tree = PrefixTreeFactory().get_prefix_tree(self.DATA_PATH.joinpath("traces"),
                                                          self.config_file_path,
                                                          unique_graph=unique_graph,
                                                          remove_trivial_loops=remove_trivial_loops)

        PrefixTreeFactory().pickle_tree(prefix_tree, PROJECT_ROOT.joinpath("resources/prefix_tree.pickle"))
        if visualize_tree:
            Visualizer(prefix_tree).visualize(name_tree)

        self.__create_positive_traces(logs, positive_traces_amount)
        self.__create_negative_traces(training_set, negative_traces_amount, syntax_tree, prefix_tree)

    def __create_positive_traces(self, traces_paths: List[str], amount: int) -> None:
        # Create positive traces
        for i in range(amount):
            rand = random.randint(0, len(traces_paths) - 1)
            shutil.copy(traces_paths[rand],
                        self.DATA_PATH.joinpath(
                            f"positive_traces/positive_xx{i}"))
            traces_paths.remove(traces_paths[rand])

    def __create_negative_traces(self, traces_paths: List[str],
                                 amount: int,
                                 syntax_tree: SyntaxTree,
                                 prefix_tree: PrefixTree) -> None:
        ls = LogScrambler(prefix_tree, syntax_tree)
        ls.get_negative_traces(amount, traces_paths)

    @staticmethod
    def get_evaluation_traces(syntax_tree: SyntaxTree,
                              dataset_path: str) -> \
            Tuple[List[List[str]], List[List[str]]]:
        positive_traces = DatasetFactory.__parse_traces(
            Path(dataset_path).joinpath("positive_traces"), syntax_tree)
        negative_traces = DatasetFactory.__parse_traces(
            Path(dataset_path).joinpath("negative_traces"), syntax_tree)

        return positive_traces, negative_traces

    @staticmethod
    def __parse_traces(path: Path, syntax_tree: SyntaxTree):
        files = os.listdir(path)
        files = [path.joinpath(line) for line in files]

        traces = []

        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [syntax_tree.search(line).name for line in lines]

                traces.append(lines)
        return traces
