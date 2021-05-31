import os
import sys

from tqdm import tqdm

from whatthelog.prefixtree.graph import Graph
from whatthelog.auto_printer import AutoPrinter


class Evaluator(AutoPrinter):
    """
    Class containing methods for evaluating state models.
    """

    pool_size_default = 16

    def __init__(self,
                 model: Graph,
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

    def update(self, new_model: Graph):
        """
        Updates the model to test
        """
        self.model = new_model

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
            self.print("Calculating specificity...")

        tn = self.process_traces(self.negative_traces_dir, self.model, debug)
        fp = len(os.listdir(self.negative_traces_dir)) - tn

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
            self.print("Calculating recall...")

        tp = self.process_traces(self.positive_traces_dir, self.model, debug)
        fn = len(os.listdir(self.positive_traces_dir)) - tp

        # Calculate the final result
        res: float = tp / (tp + fn)

        return res

    @staticmethod
    def process_traces(trace_dir: str, model: Graph, debug: bool = False) -> int:

        count = 0
        for filename in tqdm(os.listdir(trace_dir), file=sys.stdout, leave=False, disable=not debug):

            # Open the file
            with open(os.path.join(trace_dir, filename), 'r') as f:
                if model.match_trace(f.readlines()):
                    count += 1

        return count

    @staticmethod
    def match_trace(model: Graph, filename: str) -> bool:

        with open(filename, 'r') as f:
            return model.match_trace(f.readlines()) is not None
