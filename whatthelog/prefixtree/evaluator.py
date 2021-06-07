import os
from typing import List, Tuple

from scripts.match_trace import match_trace
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class Evaluator:
    """
    Class containing methods for evaluating state models.
    """

    def __init__(self,
                 model: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str,
                 initial_size: int = None,
                 weight_accuracy: float = 0.5,
                 weight_size: float = 0.5):

        self.model = model
        self.syntax_tree = syntax_tree
        self.positive_traces_dir = positive_traces_dir
        self.negative_traces_dir = negative_traces_dir
        self.initial_model_size = len(model) if initial_size is None else initial_size
        self.weight_accuracy = weight_accuracy
        self.weight_size = weight_size

    def update(self, new_model: Graph):
        """
        Updates the model to test
        """
        self.model = new_model

    def evaluate_accuracy(self, debug=False, minimizing: bool = True,
                          true_trees: List[PrefixTree] = None,
                          false_trees: List[PrefixTree] = None) -> Tuple[float, float, float]:
        """
        Statically evaluates a model in terms of specificity and recall. The returned
        Value is the BCR (binary classification rate) defined as `(accuracy + recall) / 2`
        :param debug: Whether or not to print debug information to the console.
        :param minimizing: Whether or not a better model should yield a lower value.
        """
        specificity = self.calc_specificity(debug=debug, minimizing=minimizing) \
            if false_trees is None else self.calc_specificity_trees(minimizing=minimizing, tree_traces=false_trees)
        recall = self.calc_recall(debug=debug, minimizing=minimizing) \
            if true_trees is None else self.calc_recall_trees(minimizing=minimizing, tree_traces=true_trees)

        return (specificity + recall) / 2, specificity, recall

    def evaluate_size(self, minimizing: bool = True) -> float:
        """
        Evaluates a model in terms of its size. The result is normalized by dividing by the initial model size.
        """

        res: float = 1 - len(self.model) / self.initial_model_size

        if minimizing:
            res = 1 - res

        return res

    def evaluate(self,
                 w_accuracy: float = None,
                 w_size: float = None,
                 minimizing: bool = True,
                 true_trees: List[PrefixTree] = None,
                 false_trees: List[PrefixTree] = None) -> Tuple[float, float, float, float]:
        """
        Evaluates the current model as a weighted sum between the size and the accuracy.
        :param w_accuracy: The weight of the relative accuracy evaluation in the final evaluation.
        :param w_size: The weight of the relative size evaluation in the final evaluation.
        :param minimizing: Whether or not a better model should yield a lower value.
        """
        if w_accuracy is None:
            w_accuracy = self.weight_accuracy

        if w_size is None:
            w_size = self.weight_size

        # Get the the accuracy
        accuracy, s, r = self.evaluate_accuracy(minimizing=minimizing, true_trees=true_trees, false_trees=false_trees)

        # Get the size
        size: float = self.evaluate_size(minimizing=minimizing)

        # Compute the final result using weights
        return w_accuracy * accuracy + w_size * size, size, s, r

    def calc_specificity(self, debug=False, minimizing: bool = True) -> float:
        """
        Calculates the specificity of a model on a given directory of traces.
        Specificity is defined as |TN| / (|TN| + |FP|),
         Where TN = True Negative and FP = False Positive.
        :param debug: Whether or not to print debug information to the console.
        :param minimizing: Whether or not a better model should yield a lower value.
        """

        # Initialize counters
        tn: int = 0
        fp: int = 0

        # Check if directory exists
        if not os.path.isdir(self.negative_traces_dir):
            raise NotADirectoryError("Log directory not found!")

        # For each file in the directory
        for filename in os.listdir(self.negative_traces_dir):

            # Open the file
            with open(os.path.join(self.negative_traces_dir, filename), 'r') as f:

                if debug:
                    print(f"Opening file {filename} to evaluate specificity...")

                # If the state model accepts the trace
                if match_trace(self.model, f.readlines(), self.syntax_tree):

                    # Increase the false positives by one, should have been rejected
                    fp += 1

                    if debug:
                        print("File incorrectly accepted")

                # If the state model rejects the trace
                else:

                    # Increase the true negatives by one, correctly rejected
                    tn += 1

                    if debug:
                        print("File correctly rejected")

        # Calculate the final result
        res: float = tn / (tn + fp)

        if minimizing:
            res = 1 - res

        if debug:
            print(f"Final specificity score: {res}")

        return res

    def calc_specificity_trees(self, tree_traces: List[PrefixTree], minimizing: bool = False):
        # Initialize counters
        tn: int = 0
        fp: int = 0

        for trace in tree_traces:
            if self.model.matches(trace):
                fp += 1
            else:
                tn += 1

        # Calculate the final result
        res: float = tn / (tn + fp)

        if minimizing:
            res = 1 - res

        return res

    def calc_recall(self, debug=False, minimizing: bool = False) -> float:
        """
        Calculates the recall of a model on a given directory of traces.
        Recall is defined as |TP| / (|TP| + |FN|),
         Where TP = True Positive and FN = False Negative.
        :param debug: Whether or not to print debug information to the console.
        :param minimizing: Whether or not a better model should yield a lower value.
        """

        # Initialize counters
        tp: int = 0
        fn: int = 0

        # Check if directory exists
        if not os.path.isdir(self.positive_traces_dir):
            raise NotADirectoryError("Log directory not found!")

        # For each file in the directory
        for filename in os.listdir(self.positive_traces_dir):

            # Open the file
            with open(os.path.join(self.positive_traces_dir, filename), 'r') as f:

                if debug:
                    print(f"Opening file {filename} to evaluate recall...")

                # If the state model accepts the trace
                if match_trace(self.model, f.readlines(), self.syntax_tree):

                    # Increase the true positives by one, correctly accepted
                    tp += 1

                    if debug:
                        print("File correctly accepted")

                # If the state model rejects the trace
                else:

                    # Increase the false negatives by one, should have been rejected
                    fn += 1

                    if debug:
                        print("File incorrectly rejected")

        # Calculate the final result
        res: float = tp / (tp + fn)

        if minimizing:
            res = 1 - res

        if debug:
            print(f"Final recall score: {res}")

        return res

    def calc_recall_trees(self, tree_traces: List[PrefixTree], minimizing: bool = False):
        # Initialize counters
        tp: int = 0
        fn: int = 0

        for trace in tree_traces:
            if self.model.matches(trace):
                tp += 1
            else:
                fn += 1

        # Calculate the final result
        res: float = tp / (tp + fn)

        if minimizing:
            res = 1 - res

        return res
