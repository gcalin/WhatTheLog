import os
from typing import List, Tuple

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
                 initial_size: int = None):

        self.model = model
        self.syntax_tree = syntax_tree
        self.positive_traces = positive_traces
        self.negative_traces = negative_traces
        self.initial_model_size = model.size() if initial_size is None else initial_size

    def evaluate(self) -> Tuple[float, float, float, float, float]:
        """
        Evaluates the current model as a weighted sum between the size and the accuracy.
        :param w_accuracy: The weight of the relative accuracy evaluation in the final evaluation.
        :param w_size: The weight of the relative size evaluation in the final evaluation.
        :return: precision, recall, f1_score, specificity, size
        """
        negative_results = list(map(lambda trace: self.model.match_log_template_trace(trace), self.negative_traces))
        positive_results = list(map(lambda trace: self.model.match_log_template_trace(trace), self.positive_traces))
        true_positives = sum(positive_results)
        false_positives = sum(negative_results)
        true_negatives = len(self.negative_traces) - false_positives
        false_negatives = len(self.positive_traces) - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(self.positive_traces)
        f1_score = 2 * (precision * recall) / (precision + recall)
        specificity = true_negatives / len(self.negative_traces)
        # print(self.model.size())
        # print(self.initial_model_size)
        size = 1 - (self.model.size()) / self.initial_model_size


        return precision, recall, f1_score, specificity, size


