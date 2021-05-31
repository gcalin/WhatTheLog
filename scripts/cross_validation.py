import os
from numpy import array_split, mean, std
from pathlib import Path
from random import shuffle
from typing import Tuple, List

from scripts.log_scrambler import produce_false_trace
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


def train_test_validation_split(syntax_tree: SyntaxTree,
                            logs_dir: str,
                            train_proportion: float,
                            validation_proportion: float,
                            syntax_file_path: str,
                            debug=False) -> Tuple[str, str, str, str, str, Graph]:

    assert train_proportion + validation_proportion < 1

    if debug:
        print("Entering data split phase")

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Create directory names
    train_dir: str = logs_dir + "_train"

    validation_dir: str = logs_dir + "_validation"
    validation_dir_true: str = validation_dir + "_true"
    validation_dir_false: str = validation_dir + "_false"

    test_dir: str = logs_dir + "_test"
    test_dir_true: str = test_dir + "_true"
    test_dir_false: str = test_dir + "_false"

    os.mkdir(test_dir_true)
    os.mkdir(test_dir_false)
    os.mkdir(validation_dir_true)
    os.mkdir(validation_dir_false)
    os.mkdir(train_dir)

    # Split traces
    traces: List[str] = os.listdir(logs_dir)
    train_traces_num: int = int(len(traces) * train_proportion)
    validation_traces_num: int = int(len(traces) * validation_proportion)

    train_traces = traces[:train_traces_num]
    validation_traces = traces[train_traces_num:train_traces_num + validation_traces_num]
    test_traces = traces[train_traces_num + validation_traces_num:]

    # For each train trace
    for train_trace in train_traces:

        # Create new name
        old_name = os.path.join(logs_dir, train_trace)
        new_name = os.path.join(train_dir, train_trace)

        # Move the trace
        os.rename(old_name, new_name)

    model: Graph = PrefixTreeFactory.get_prefix_tree(traces_dir=train_dir,
                                                     config_file_path=syntax_file_path, remove_trivial_loops=False)

    # For each validation trace
    for test_trace_name in validation_traces:

        # Create new name
        old_name = os.path.join(logs_dir, test_trace_name)
        new_name = os.path.join(validation_dir_true, test_trace_name)

        # Create a false trace
        produce_false_trace(old_name,
                            os.path.join(validation_dir_false, test_trace_name),
                            syntax_tree,
                            model)

        # Move the trace
        os.rename(old_name, new_name)

    # For each test trace
    for test_trace_name in test_traces:

        # Create new names
        old_name = os.path.join(logs_dir, test_trace_name)
        new_name = os.path.join(test_dir_true, test_trace_name)

        # Produce a false trace
        produce_false_trace(old_name,
                            os.path.join(test_dir_false, test_trace_name),
                            syntax_tree,
                            model)

        # Move the trace
        os.rename(old_name, new_name)

    return logs_dir, validation_dir_true, validation_dir_false, test_dir_true, test_dir_false, model


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
    :param debug: Whether or not to print debug information to the console.
    """

    # Arrays used to collect specificity and recall
    specificity, recall = [], []

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Get all traces and randomize their order
    traces: List[str] = os.listdir(logs_dir)
    shuffle(traces)

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(traces, k)

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
            new_name_specificity = os.path.join(temp_dir_name_specificity, trace_name + "_false")
            produce_false_trace(os.path.join(temp_dir_name_recall, trace_name), new_name_specificity, syntax_tree, model)

        # Evaluate the model and append the results
        evaluator: Evaluator = Evaluator(model, syntax_tree, str(temp_dir_name_recall), str(temp_dir_name_specificity))
        s = evaluator.calc_specificity(debug=debug)
        r = evaluator.calc_recall(debug=debug)
        specificity.append(s)
        recall.append(r)

        if debug:
            print(f"Iteration {count + 1} of {k} done: speci. is {s}, recall is {r}.")
            print("Cleaning up directories...")

        # Clean up directories
        for trace_name in fold:
            old_name = os.path.join(logs_dir, trace_name)
            new_name_recall = os.path.join(temp_dir_name_recall, trace_name)
            new_name_specificity = os.path.join(temp_dir_name_specificity, trace_name + "_false")

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


# Example usage
if __name__ == '__main__':

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent

    cfg_file = str(project_root.joinpath('resources/config.json'))
    true_traces = str(project_root.joinpath('tests/resources/truelogs'))
    false_traces = str(project_root.joinpath('tests/resources/testlogs'))

    k_fold_cross_validation(
        SyntaxTreeFactory().parse_file(cfg_file),
        false_traces,
        8,
        debug=True
    )
