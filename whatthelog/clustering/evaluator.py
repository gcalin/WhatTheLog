import os
import sys
from typing import List

from tqdm import tqdm

from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.prefixtree.matchable_graph import MatchableGraph
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import UnidentifiedLogException


class Evaluator(AutoPrinter):
    """
    Class containing methods for evaluating state models.
    """

    pool_size_default = 16

    def __init__(self,
                 model: MatchableGraph,
                 positive_traces_dir: str,
                 negative_traces_dir: str,
                 initial_size: int = None,
                 weight_accuracy: float = 0.5,
                 weight_size: float = 0.5):

        self.model = model
        self.positive_traces_dir = positive_traces_dir
        self.negative_traces_dir = negative_traces_dir
        self.initial_model_size = len(model) if initial_size is None else initial_size
        self.weight_accuracy = weight_accuracy
        self.weight_size = weight_size
        self.positive_templates = None
        self.negative_templates = None

    def update(self, new_model: MatchableGraph):
        """
        Updates the model to test
        """
        self.model = new_model

    def build_cache(self, force_rebuild: bool = False, debug: bool = False):
        """
        Processes the trace files and stores them in memory as lists of templates
        in order to speed-up evaluation.

        :param force_rebuild: if True forces the rebuilding of the cache
                              even if a previous cache is already built.
        :param debug: if True enables printing logs and progressbar to the console.
        """

        assert os.path.isdir(self.positive_traces_dir), f"Invalid directory for positive traces! " \
                                                        f"Directory {self.positive_traces_dir} not found."

        syntax_tree: SyntaxTree = self.model.syntax_tree

        # --- Check if positive traces cache already exists or force rebuild enabled ---
        if debug: self.print("Building cache of positive traces...")
        if not self.positive_templates or force_rebuild:

            self.positive_templates: List[List[str]] = []

            # --- Iter files ---
            for filename in tqdm(os.listdir(self.positive_traces_dir),
                                 file=sys.stdout, leave=False, disable=not debug):

                with open(os.path.join(self.positive_traces_dir, filename), 'r') as f:

                    trace: List[str] = []
                    lines: List[str] = f.readlines()

                    # --- Iter lines ---
                    for line in lines:

                        # --- Find line template name or throw exception if not found ---
                        name: str = syntax_tree.search(line).name
                        if not name:
                            raise UnidentifiedLogException()

                        trace.append(name)

                    self.positive_templates.append(trace)

        assert os.path.isdir(self.negative_traces_dir), f"Invalid directory for negative traces! " \
                                                        f"Directory {self.negative_traces_dir} not found."

        # --- Check if negative traces cache already exists or force rebuild enabled ---
        if debug: self.print("Building cache of negative traces...")
        if not self.negative_templates or force_rebuild:

            self.negative_templates: List[List[str]] = []

            # --- Iter files ---
            for filename in tqdm(os.listdir(self.negative_traces_dir),
                                 file=sys.stdout, leave=False, disable=not debug):

                with open(os.path.join(self.negative_traces_dir, filename), 'r') as f:

                    trace: List[str] = []
                    lines: List[str] = f.readlines()

                    # --- Iter lines ---
                    for line in lines:

                        # --- Find line template name or throw exception if not found ---
                        name: str = syntax_tree.search(line).name
                        if not name:
                            raise UnidentifiedLogException()

                        trace.append(name)

                    self.negative_templates.append(trace)

    def evaluate_accuracy(self, debug=False) -> float:
        """
        Statically evaluates a model in terms of specificity and recall. The returned
        Value is the BCR (binary classification rate) defined as `(accuracy + recall) / 2`
        :param debug: Whether or not to print debug information to the console.
        """

        specificity = self.calc_specificity(debug=debug)
        recall = self.calc_recall(debug=debug)

        return (specificity + recall) / 2

    def evaluate_size(self) -> float:
        """
        Evaluates a model in terms of its size. The result is normalized by dividing by the initial model size.
        """
        return 1 - len(self.model) / self.initial_model_size

    def evaluate(self,
                 w_accuracy: float = None,
                 w_size: float = None) -> float:
        """
        Evaluates the current model as a weighted sum between the size and the accuracy.
        :param w_accuracy: The weight of the relative accuracy evaluation in the final evaluation.
        :param w_size: The weight of the relative size evaluation in the final evaluation.
        """

        if w_accuracy is None:
            w_accuracy = self.weight_accuracy

        if w_size is None:
            w_size = self.weight_size

        # Get the the accuracy
        accuracy: float = self.evaluate_accuracy()

        # Get the size
        size: float = self.evaluate_size()

        # Compute the final result using weights
        return w_accuracy * accuracy + w_size * size

    def calc_specificity(self, debug=False) -> float:
        """
        Calculates the specificity of a model on a given directory of traces.
        Specificity is defined as |TN| / (|TN| + |FP|),
         Where TN = True Negative and FP = False Positive.
        :param debug: Whether or not to print debug information to the console.
        """

        # Check if directory exists
        if not os.path.isdir(self.negative_traces_dir):
            raise NotADirectoryError("Log directory not found!")

        if debug:
            self.print(f"Calculating specificity for {len(os.listdir(self.negative_traces_dir))} test traces...")

        fp = self.process_templates(self.negative_templates, self.model, debug) \
            if self.negative_templates \
            else self.process_traces(self.negative_traces_dir, self.model, debug)
        tn = len(os.listdir(self.negative_traces_dir)) - fp

        # Calculate the final result
        res: float = tn / (tn + fp)

        return res

    def calc_recall(self, debug=False) -> float:
        """
        Calculates the recall of a model on a given directory of traces.
        Recall is defined as |TP| / (|TP| + |FN|),
         Where TP = True Positive and FN = False Negative.
        :param debug: Whether or not to print debug information to the console.
        """

        # Check if directory exists
        if not os.path.isdir(self.positive_traces_dir):
            raise NotADirectoryError("Log directory not found!")

        if debug:
            self.print(f"Calculating recall for {len(os.listdir(self.positive_traces_dir))} test traces......")

        tp = self.process_templates(self.positive_templates, self.model, debug) \
            if self.positive_templates \
            else self.process_traces(self.positive_traces_dir, self.model, debug)
        fn = len(os.listdir(self.positive_traces_dir)) - tp

        # Calculate the final result
        res: float = tp / (tp + fn)

        return res

    @staticmethod
    def process_traces(trace_dir: str, model: MatchableGraph, debug: bool = False) -> int:

        count = 0
        for filename in tqdm(os.listdir(trace_dir), file=sys.stdout, leave=False, disable=not debug):

            # Open the file
            with open(os.path.join(trace_dir, filename), 'r') as f:
                if model.match_trace(f.readlines()):
                    count += 1

        return count

    @staticmethod
    def process_templates(traces: List[List[str]], model: MatchableGraph, debug: bool = False) -> int:

        count = 0
        for trace in tqdm(traces, file=sys.stdout, leave=False, disable=not debug):

            if model.match_templates(trace, debug):
                count += 1

        return count

    @staticmethod
    def match_trace(model: MatchableGraph, filename: str) -> bool:

        with open(filename, 'r') as f:
            return model.match_trace(f.readlines()) is not None
