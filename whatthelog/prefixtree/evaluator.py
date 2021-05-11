import os


from scripts.match_trace import match_trace
from whatthelog.prefixtree.graph import Graph
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class Evaluator:
    """
    Class containing methods for evaluating state models.
    """

    def __init__(self,
                 model: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str):

        self.model = model
        self.syntax_tree = syntax_tree
        self.positive_traces_dir = positive_traces_dir
        self.negative_traces_dir = negative_traces_dir

    def update(self, new_model: Graph):
        """
        Updates the model to test
        """
        self.model = new_model

    def evaluate(self, debug=False) -> float:
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
        Evaluates a model in terms of its size.
        """
        return (len(self.model) + len(self.model.edges)) / 2

    def evaluate_relative(self,
                          other_model: Graph,
                          w_accuracy: float = 1,
                          w_size: float = 0,
                          minimizing: bool = False) -> float:
        """
        Evaluates the current model relatively against a different model.
        :param other_model: The model to compare against.
        :param w_accuracy: The weight of the relative accuracy evaluation in the final evaluation.
        :param w_size: The weight of the relative size evaluation in the final evaluation.
        :param minimizing: Whether or not to the resulting value should be minimizing or not.
        """

        # Construct an evaluator for the other model
        other: Evaluator = Evaluator(other_model, self.syntax_tree, self.positive_traces_dir, self.negative_traces_dir)

        # Get the static accuracies
        accuracy_before: float = self.evaluate()
        accuracy_after: float = other.evaluate()

        # Get the static sizes
        size_before = self.evaluate_size()
        size_after = other.evaluate_size()

        # Compute the differences for each evaluation
        accuracy_dif = (accuracy_after - accuracy_before) / 2
        size_dif = (size_after - size_before) / 2

        # Compute the final result using weights
        return (w_accuracy * accuracy_dif + w_size * size_dif) * -1 if not minimizing else 1

    def calc_specificity(self, debug=False) -> float:
        """
        Calculates the specificity of a model on a given directory of traces.
        Specificity is defined as |TN| / (|TN| + |FP|),
         Where TN = True Negative and FP = False Positive.
        :param debug: Whether or not to print debug information to the console.
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

        if debug:
            print(f"Final recall score: {res}")

        return res
