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
import numpy as np
from pathlib import Path
import sys
from sknetwork.utils import edgelist2adjacency
from sknetwork.hierarchy import LouvainHierarchy
from tqdm import tqdm
from time import time
from typing import Tuple, List

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.adjacency_graph import AdjacencyGraph
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

    def __init__(self, tree: PrefixTree,
                 true_traces_dir: str = None,
                 false_traces_dir: str = None,
                 skip_cache_build: bool = False):

        if not true_traces_dir:
            self.true_traces_dir = project_root.joinpath(self.default_true_traces_dir)
        else:
            self.true_traces_dir = true_traces_dir
        if not false_traces_dir:
            self.false_traces_dir = project_root.joinpath(self.default_false_traces_dir)
        else:
            self.false_traces_dir = false_traces_dir

        self.tree = tree
        self.evaluator = Evaluator(tree, self.true_traces_dir, self.false_traces_dir)
        if not skip_cache_build:
            self.evaluator.build_cache(debug=True)

    def get_dendrogram(self) -> np.ndarray:
        """
        Builds and returns the dendrogram of the base tree using the Louvain algorithm.
        :return: a 2d array representing the dendrogram
        """

        model = LouvainHierarchy()
        adj_list = self.tree.get_adj_list(remove_self_loops=True)
        adjacency = edgelist2adjacency(adj_list)

        self.print(f"Running Louvain clustering algorithm...")
        return model.fit_transform(adjacency)

    def eval_merges(self, dendrogram: np.ndarray, n_merges: int, step: int = 1, debug: bool = False)\
            -> List[Tuple[float, float, float]]:
        """
        Runs the given amount of merges on the graph, based on the given dendrogram,
        evaluating the model every 'step' merges.
        Returns a list of evaluation results, where every result is a tuple of:
            evaluation time, specificity, recall
        :param dendrogram: the input dendrogram to use for merging
        :param n_merges: the number of merges to perform
        :param step: the period to use between each evaluation. 1 means every cycle.
        :param debug: if True enables logging
        :return: the list of evaluation results
        """

        if debug: self.print(f"Running {n_merges} merges...")
        merges = self.relabel_dendrogram(dendrogram)
        accuracies = []
        start_idx = self.tree.get_state_index_by_id(id(self.tree.start_node))
        pbar = tqdm(range(n_merges), file=sys.stdout, leave=False, disable = not debug)
        for i in pbar:

            if i >= len(merges): break

            dest, source = merges[i]
            if dest not in self.tree or source not in self.tree:
                continue
            if dest == start_idx or source == start_idx:
                continue

            self.tree.full_merge_states(self.tree.states[dest], self.tree.states[source])

            if i % step == 0:
                start_time = time()
                specificity, recall = self.eval_model(self.evaluator)
                time_delta = round(time() - start_time, 5)
                accuracies.append((time_delta, specificity, recall))

        return accuracies

    def merge_full(self, dendrogram: np.ndarray):

        merges = self.relabel_dendrogram(dendrogram)
        self.print(f"Running {len(merges)} merges...")
        start_idx = self.tree.get_state_index_by_id(id(self.tree.start_node))
        pbar = tqdm(merges, file=sys.stdout, leave=False)
        for merge in pbar:

            dest, source = merge
            if dest not in self.tree or source not in self.tree:
                continue
            if dest == start_idx or source == start_idx:
                continue

            self.tree.full_merge_states(self.tree.states[dest], self.tree.states[source])

    def relabel_dendrogram(self, dendrogram: np.ndarray) -> list:

        # Relabel nodes in the dendrogram
        length = len(self.tree)
        merged_states = {}
        merges = []
        self.print("Relabeling merges in dendrogram...")
        for count, merge in enumerate(dendrogram):

            merge = merge.tolist()
            dest, source = int(merge[0]), int(merge[1])
            dest_index = dest if dest < length else merged_states[dest]
            source_index = source if source < length else merged_states[source]

            merges.append((dest_index, source_index))
            merged_states[length + count] = dest_index

        return merges

    @staticmethod
    def eval_model(evaluator: Evaluator, debug: bool = False) -> Tuple[float, float]:
        """
        Worker function to evaluate specificity, recall and size of the given model.
        Intended to be used in parallel subprocesses.
        :param evaluator: the Evaluator instance to use for evaluation
        :param debug: if True enables logging
        :return: a tuple of specificity and recall
        """

        specificity = evaluator.calc_specificity(debug)
        recall = evaluator.calc_recall(debug)
        return specificity, recall

    @staticmethod
    def pickle_model(model: AdjacencyGraph, file: str) -> None:
        """
        Pickles and dumps the given model instance into a given file.
        If the file does not exist it will be created.
        :param model: the model instance to pickle
        :param file: the file to dump the pickled model to
        """

        with open(file, 'wb+') as f:
            pickle.dump(model, f)

    @staticmethod
    def unpickle_model(file: str) -> AdjacencyGraph:
        """
        Parses a pickled model instance from a file.
        :param file: the pickle file representing the instance
        :return: the parsed StateGraph instance
        """

        if not os.path.isfile(file):
            raise FileNotFoundError("Pickle file not found!")

        with open(file, 'rb') as f:
            model = pickle.load(f)

        return model
