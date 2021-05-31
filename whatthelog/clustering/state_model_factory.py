# -*- coding: utf-8 -*-
"""
Created on Tuesday 05/25/2021
Author: Tommaso Brandirali
Email: tommaso.brandirali@gmail.com
"""

#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pickle
from copy import deepcopy
from math import floor
import numpy as np
from pathlib import Path
import sys
from sknetwork.utils import edgelist2adjacency
from sknetwork.hierarchy import LouvainHierarchy, Paris
from tqdm import tqdm
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, wait

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.auto_printer import AutoPrinter
from whatthelog.clustering.evaluator import Evaluator

project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


#****************************************************************************************************
# State Model Factory
#****************************************************************************************************

class StateModelFactory(AutoPrinter):
    """
    State model factory class.
    """

    default_true_traces_dir = "out/true_traces"
    default_false_traces_dir = "out/false_traces"
    default_pool_size = 8
    algorithms = {
        'louvain': LouvainHierarchy,
        'paris': Paris
    }

    def __init__(self, tree: PrefixTree, true_traces_dir: str = None, false_traces_dir: str = None):

        if not true_traces_dir:
            self.true_traces_dir = project_root.joinpath(self.default_true_traces_dir)
        if not false_traces_dir:
            self.false_traces_dir = project_root.joinpath(self.default_false_traces_dir)

        self.tree = tree
        self.evaluator = Evaluator(tree, self.true_traces_dir, self.false_traces_dir)

    def run_clustering(self, algorithm: str = 'louvain') -> Tuple[Graph, List[Tuple[float, float]]]:
        """
        Performs a clustering of the prefix tree using the given hierarchical algorithm,
        returns the resulting state model as a Graph of merged states.

        :param algorithm: the algorithm to use for clustering
        :return: the reduced prefix tree state model
        """

        model = self.algorithms[algorithm]()
        adj_list = self.tree.get_adj_list(remove_self_loops=False)
        adjacency = edgelist2adjacency(adj_list)

        self.print(f"Running '{algorithm}' clustering algorithm...")
        dendrogram = model.fit_transform(adjacency)

        return self.build_from_dendrogram(dendrogram)

    def build_from_dendrogram(self, dendrogram: np.ndarray) -> Tuple[Graph, List[Tuple[float, float]]]:
        """
        Creates a state model by recursively merging the Prefix Tree nodes
        according to the given dendrogram produced by hierarchical clustering.

        :param dendrogram: the input dendrogram
        :return: the resulting Graph instance
        """

        tree = deepcopy(self.tree)
        length = len(tree)
        merged_states = {}
        futures = []

        # TODO: use evaluator to merge until threshold fitness
        self.print("Merging states...")
        with ProcessPoolExecutor(self.default_pool_size) as pool:
            pbar = tqdm(dendrogram, file=sys.stdout, leave=False)
            for count, merge in enumerate(pbar):

                merge = merge.tolist()
                dest, source = int(merge[0]), int(merge[1])
                dest_index = dest if dest < length else merged_states[dest]
                source_index = source if source < length else merged_states[source]

                tree.merge_states(tree.states[dest_index], tree.states[source_index])
                merged_states[length + count] = dest_index

                if count % 100 == 0:
                    copy = deepcopy(tree)
                    futures.append(pool.submit(self.eval_model, copy))

        self.print("Waiting for evaluation to complete...")
        accuracies = []
        for x in tqdm(futures, file=sys.stdout, leave=False):
            accuracies.append(x.result())

        return tree, accuracies

    def eval_model(self, model: Graph) -> Tuple[float, float]:
        """
        Worker function to evaluate specificity, recall and size of the given model.
        Intended to be used in parallel subprocesses.
        :param model: the current model to evaluate.
        :return: a tuple of specificity and recall
        """

        evaluator = Evaluator(model, self.true_traces_dir, self.false_traces_dir)
        specificity = evaluator.calc_specificity()
        recall = evaluator.calc_recall()
        return specificity, recall

    @staticmethod
    def pickle_model(model: Graph, file: str) -> None:
        """
        Pickles and dumps the given model instance into a given file.
        If the file does not exist it will be created.
        :param model: the model instance to pickle
        :param file: the file to dump the pickled model to
        """

        with open(file, 'wb+') as f:
            pickle.dump(model, f)

    @staticmethod
    def unpickle_model(file: str) -> Graph:
        """
        Parses a pickled model instance from a file.
        :param file: the pickle file representing the instance
        :return: the parsed Graph instance
        """

        if not os.path.isfile(file):
            raise FileNotFoundError("Pickle file not found!")

        with open(file, 'rb') as f:
            model = pickle.load(f)

        return model
