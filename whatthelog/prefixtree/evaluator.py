import os
from typing import List

from whatthelog.prefixtree.graph import Graph
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class Evaluator:
    """
    Class containing methods for evaluating state models.
    """

    def __init__(self,
                 model: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces: List[List[str]],
                 negative_traces: List[List[str]],
                 initial_size: int = None,
                 weight_accuracy: float = 0.5,
                 weight_size: float = 0.5):

        self.model = model
        self.syntax_tree = syntax_tree
        self.positive_traces = positive_traces
        self.negative_traces = negative_traces
        self.initial_model_size = model.size() if initial_size is None else initial_size
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
        # accuracy: float = self.evaluate_accuracy()
        # # Get the size
        # size: float = self.evaluate_size()
        #
        # # Compute the final result using weights
        # return w_accuracy * accuracy + w_size * size
        return self.calc_recall()

    def calc_specificity(self, debug=False) -> float:
        """
        Calculates the specificity of a model on a given directory of traces.
        Specificity is defined as |TN| / (|TN| + |FP|),
         Where TN = True Negative and FP = False Positive.
        :param debug: Whether or not to print debug information to the console.
        """

        results = list(map(lambda trace: self.model.match_log_template_trace(trace), self.negative_traces))

        res: float = (len(self.negative_traces) - sum(results)) / len(self.negative_traces)

        if debug:
            print(f"Final specificity score: {res}")

        return res

    def calc_recall(self, debug=False) -> float:
        """
        Calculates the recall of a model on a given directory of traces.
        Recall is defined as |TP| / (|TP| + |FN|),
         Where TP = True Positive and FN = False Negative.
        :param debug: Whether or not to print debug information to the console.
        """
        results = list(map(lambda trace: self.model.match_log_template_trace(trace), self.positive_traces))

        # Calculate the final result
        res: float = sum(results) / len(self.positive_traces)

        if debug:
            print(f"Final recall score: {res}")

        return res
