import os
from pathlib import Path
from typing import Tuple

from scripts.match_trace import match_trace
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


def evaluate_accuracy(syntax_tree: SyntaxTree,
                      state_model: PrefixTree,
                      positive_logs_dir: str,
                      faulty_logs_dir: str,
                      debug=False) -> Tuple[float, float]:
    """
    Calculates the specificity and recall of a state model.
    :param syntax_tree: The syntax tree used to validate individual logs.
    :param state_model: The state model that accepts or rejects a given trace.
    :param positive_logs_dir: The directory containing the positive log traces used to test recall.
    :param faulty_logs_dir: The directory containing the negative log traces used to test specificity.
    :param debug: Whether or not to print debug information to the console.
    """

    # Calculate specificity
    specificity = calc_specificity(syntax_tree, state_model, faulty_logs_dir, debug)

    # Calculate recall
    recall = calc_recall(syntax_tree, state_model, positive_logs_dir, debug)

    return specificity, recall


def calc_specificity(syntax_tree: SyntaxTree,
                     state_model: PrefixTree,
                     faulty_logs_dir: str,
                     debug=False) -> float:
    """
    Calculates the specificity of a model on a given directory of traces.
    Specificity is defined as |TN| / (|TN| + |FP|),
     Where TN = True Negative and FP = False Positive.
    :param syntax_tree: The syntax tree used to validate individual logs.
    :param state_model: The state model that accepts or rejects a given trace.
    :param faulty_logs_dir: The directory containing the negative log traces.
    :param debug: Whether or not to print debug information to the console.
    """

    # Initialize counters
    tn: int = 0
    fp: int = 0

    # Check if directory exists
    if not os.path.isdir(faulty_logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # For each file in the directory
    for filename in os.listdir(faulty_logs_dir):

        # Open the file
        with open(os.path.join(faulty_logs_dir, filename), 'r') as f:

            if debug:
                print(f"Opening file {filename} to evaluate specificity...")

            # If the state model accepts the trace
            if match_trace(state_model, f.readlines(), syntax_tree):

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


def calc_recall(syntax_tree: SyntaxTree,
                state_model: PrefixTree,
                positive_logs_dir: str,
                debug=False) -> float:
    """
    Calculates the recall of a model on a given directory of traces.
    Recall is defined as |TP| / (|TP| + |FN|),
     Where TP = True Positive and FN = False Negative.
    :param syntax_tree: The syntax tree used to validate individual logs.
    :param state_model: The state model that accepts or rejects a given trace.
    :param positive_logs_dir: The directory containing the negative log traces.
    :param debug: Whether or not to print debug information to the console.
    """

    # Initialize counters
    tp: int = 0
    fn: int = 0

    # Check if directory exists
    if not os.path.isdir(positive_logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # For each file in the directory
    for filename in os.listdir(positive_logs_dir):

        # Open the file
        with open(os.path.join(positive_logs_dir, filename), 'r') as f:

            if debug:
                print(f"Opening file {filename} to evaluate recall...")

            # If the state model accepts the trace
            if match_trace(state_model, f.readlines(), syntax_tree):

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


if __name__ == '__main__':
    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent

    cfg_file = str(project_root.joinpath('resources/config.json'))
    true_traces = str(project_root.joinpath('tests/resources/true_logs'))
    false_traces = str(project_root.joinpath('tests/resources/testlogs'))

    st = Parser().parse_file(cfg_file)
    pt = PrefixTreeFactory.get_prefix_tree(true_traces, cfg_file)

    evaluate_accuracy(st, pt, true_traces, false_traces, True)
