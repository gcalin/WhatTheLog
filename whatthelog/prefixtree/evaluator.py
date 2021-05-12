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
                 negative_traces_dir: str,
                 initial_size: int = None):

        self.model = model
        self.syntax_tree = syntax_tree
        self.positive_traces_dir = positive_traces_dir
        self.negative_traces_dir = negative_traces_dir
        self.initial_model_size = len(model) if initial_size is None else initial_size

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
        return len(self.model) / self.initial_model_size

    def evaluate(self,
                 w_accuracy: float = 0.5,
                 w_size: float = 0.5) -> float:
        """
        Evaluates the current model as a weighted sum between the size and the accuracy.
        :param w_accuracy: The weight of the relative accuracy evaluation in the final evaluation.
        :param w_size: The weight of the relative size evaluation in the final evaluation.
        """
        # Get the the accuracy
        accuracy: float = self.evaluate()

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
