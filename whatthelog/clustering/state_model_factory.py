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
from copy import deepcopy, copy
import numpy as np
from pathlib import Path
import sys
from sknetwork.utils import edgelist2adjacency
from sknetwork.hierarchy import LouvainHierarchy, Paris, BaseHierarchy
from tqdm import tqdm
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        self.evaluator.build_cache(debug=True)

    def run_clustering(self, algorithm: str = 'louvain') -> Tuple[AdjacencyGraph, List[Tuple[float, float]]]:
        """
        Performs a clustering of the prefix tree using the given hierarchical algorithm,
        returns the resulting state model as a StateGraph of merged states.

        :param algorithm: the algorithm to use for clustering
        :return: the reduced prefix tree state model
        """

        model = self.algorithms[algorithm]()
        dendrogram = self.get_dendrogram(model)

        return self.build_from_dendrogram_parallel_eval(dendrogram)

    def get_dendrogram(self, model: BaseHierarchy) -> np.ndarray:
        """
        Builds and returns the dendrogram of the base tree using the input algorithm.
        :param model: the sknetwork hierarchical algorithm to use
        :return: a 2d array representing the dendrogram
        """

        adj_list = self.tree.get_adj_list(remove_self_loops=True)
        adjacency = edgelist2adjacency(adj_list)

        self.print(f"Running clustering algorithm...")
        return model.fit_transform(adjacency)

    def build_from_dendrogram_parallel_eval(self, dendrogram: np.ndarray)\
            -> Tuple[AdjacencyGraph, List[Tuple[float, float]]]:
        """
        Creates a state model by recursively merging the Prefix Tree nodes
        according to the given dendrogram produced by hierarchical clustering.

        Evaluates the accuracies every 200 merges in parallel processes.

        :param dendrogram: the input dendrogram
        :return: the resulting StateGraph instance and the list of evaluation results
        """

        tree = deepcopy(self.tree)
        start_idx = tree.state_indices_by_id[id(tree.start_node)]
        length = len(tree)
        futures = []
        accuracies = []

        # Relabel nodes in the dendrogram
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

        self.print(f"Merging {len(merges)} states...")
        with ProcessPoolExecutor(self.default_pool_size) as pool:
            pbar = tqdm(merges, file=sys.stdout, leave=False)
            for count, merge in enumerate(pbar):

                dest, source = merge
                # self.print(f"Merging {source} into {dest}")
                if dest not in tree or source not in tree:
                    continue

                if dest == start_idx or source == start_idx:
                    continue

                if tree.states[dest].is_terminal or tree.states[source].is_terminal:
                    continue

                tree.full_merge_states(tree.states[dest], tree.states[source])

                if count % 100 == 0:
                    tree_copy: AdjacencyGraph = deepcopy(tree)
                    evaluator_copy: Evaluator = copy(self.evaluator)
                    evaluator_copy.update(tree_copy)
                    futures.append(pool.submit(self.eval_model, evaluator_copy))

            self.print(f"Completing evaluation of {len(futures)} futures...")
            for _ in tqdm(as_completed(futures), total=len(futures), file=sys.stdout, leave=False):
                pass

        for x in futures:
            accuracies.append(x.result())

        return tree, accuracies

    def build_from_dendrogram_linear_eval(self, dendrogram: np.ndarray) \
            -> Tuple[AdjacencyGraph, List[Tuple[float, float]]]:
        """
        Creates a state model by recursively merging the Prefix Tree nodes
        according to the given dendrogram produced by hierarchical clustering.

        Evaluates the accuracies every 200 merges.

        :param dendrogram: the input dendrogram
        :return: the resulting StateGraph instance and the list of evaluation results
        """

        tree = deepcopy(self.tree)
        length = len(tree)
        merged_states = {}
        accuracies = []

        # TODO: use evaluator to merge until threshold fitness
        self.print("Merging states...")
        pbar = tqdm(dendrogram, file=sys.stdout, leave=False)
        for count, merge in enumerate(pbar):

            merge = merge.tolist()
            dest, source = int(merge[0]), int(merge[1])
            dest_index = dest if dest < length else merged_states[dest]
            source_index = source if source < length else merged_states[source]

            tree.merge_states(tree.states[dest_index], tree.states[source_index])
            merged_states[length + count] = dest_index

            if count % 200 == 0:
                self.evaluator.update(tree)
                accuracies.append(self.eval_model(self.evaluator))

        return tree, accuracies

    @staticmethod
    def eval_model(evaluator: Evaluator) -> Tuple[float, float]:
        """
        Worker function to evaluate specificity, recall and size of the given model.
        Intended to be used in parallel subprocesses.
        :param evaluator: the Evaluator instance to use for evaluation
        :return: a tuple of specificity and recall
        """

        specificity = evaluator.calc_specificity()
        recall = evaluator.calc_recall()
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
