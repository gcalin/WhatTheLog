import os
from numpy import array_split, mean, std
from pathlib import Path
from typing import Tuple, List

from scripts.log_scrambler import produce_false_trace
from scripts.match_trace import match_trace
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


def k_fold_cross_validation(syntax_tree: SyntaxTree,
                            logs_dir: str,
                            k: int,
                            config_location: str = "resources/config.json",
                            debug=False) -> Tuple[List[float], List[float]]:
    """
    Performs k-fold cross validation on a given set of traces, collecting recall and specificity.
    :param syntax_tree: The syntax tree used to match a particular log entry.
    :param logs_dir: The directory containing the traces.
    :param k: The number of folds.
    :param config_location: The location of the configuration used to build the syntax tree.
    :param debug: Whether or not to print debug infomation to the console.
    """

    # Arrays used to collect specificity and recall
    specificity, recall = [], []

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(os.listdir(logs_dir), k)

    if debug:
        print(f"Entering k-fold cross validation, with {len(os.listdir(logs_dir))} "
              f"traces split into {len(folds)} folds...")

    # For each fold
    for count, fold in enumerate(folds):

        if debug:
            print(f"Iteration {count + 1} started: creating directories...")

        # Create new directories for separating the logs
        # The current fold is moved to a new directory, which is used to test recall
        # False logs are generated into another new directory from current fold, to test specificity
        temp_dir_name_specificity: Path = Path("tmp_test_specificity_" + str(count))
        temp_dir_name_recall: Path = Path("tmp_test_recall_" + str(count))
        os.mkdir(temp_dir_name_specificity)
        os.mkdir(temp_dir_name_recall)

        if debug:
            print(f"Directories created.\nMoving current fold...")

        # For each trace in the current fold
        for trace_name in fold:
            old_name = os.path.join(logs_dir, trace_name)
            new_name_recall = os.path.join(temp_dir_name_recall, trace_name)

            # Move the trace to the recall directory such that it is
            # Not used for training
            os.rename(old_name, new_name_recall)

        if debug:
            print(f"Current fold of length {len(fold)} moved.\nBuilding the prefix tree...")

        # Build model based on all traces except the ones in the current fold
        model: PrefixTree = PrefixTreeFactory.get_prefix_tree(logs_dir, "../" + config_location)

        if debug:
            print(f"Prefix tree built.\nMinimizing the model...")

        # TODO Minimize initial model...

        if debug:
            print(f"Minimizing finished.\nEvaluating model...")

        # Create false traces based on the current fold
        for trace_name in os.listdir(temp_dir_name_recall):
            new_name_specificity = os.path.join(temp_dir_name_specificity, trace_name)
            produce_false_trace(os.path.join(temp_dir_name_recall, trace_name), new_name_specificity, syntax_tree, model)

        # Evaluate the model and append the results
        s, r = evaluate_accuracy(syntax_tree, model, str(logs_dir), str(temp_dir_name_specificity))
        specificity.append(s)
        recall.append(r)

        if debug:
            print(f"Iteration {count + 1} of {k} done: speci. is {s}, recall is {r}.")
            print("Cleaning up directories...")

        # Clean up directories
        for trace_name in fold:
            old_name = os.path.join(logs_dir, trace_name)
            new_name_recall = os.path.join(temp_dir_name_recall, trace_name)
            new_name_specificity = os.path.join(temp_dir_name_specificity, trace_name)

            # Move the trace back to the main directory from the recall directory
            os.rename(new_name_recall, old_name)

            # Remove the file from the specificity directory
            os.remove(new_name_specificity)

        # Remove the directories
        os.rmdir(temp_dir_name_recall)
        os.rmdir(temp_dir_name_specificity)

        if debug:
            print("Finished cleaning up all the directories.")

    if debug:
        print(f"Finished validation: mean speci.:{mean(specificity)} with deviation {std(specificity)},"
              f" mean recall:{mean(recall)} with deviation {std(recall)}.")

    return specificity, recall


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


# Example usage
if __name__ == '__main__':

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent

    cfg_file = str(project_root.joinpath('resources/config.json'))
    true_traces = str(project_root.joinpath('tests/resources/truelogs'))
    false_traces = str(project_root.joinpath('tests/resources/testlogs'))

    k_fold_cross_validation(
        Parser().parse_file(cfg_file),
        false_traces,
        8,
        debug=True
    )

    # st = Parser().parse_file(cfg_file)
    # pt = PrefixTreeFactory.get_prefix_tree(true_traces, cfg_file)
    #
    # evaluate_accuracy(st, pt, true_traces, false_traces, True)
