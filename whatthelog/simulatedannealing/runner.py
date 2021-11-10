import os
import time
from pathlib import Path
import random
from typing import List

import numpy as np
from numpy import array_split

from scripts.log_scrambler import produce_false_trace
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.simulatedannealing.coolingschedule import LundySchedule
from whatthelog.simulatedannealing.search import SimulatedAnnealing
from whatthelog.simulatedannealing.selection import TournamentSelection
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


def kfcv_accuracy(syntax_tree: SyntaxTree,
                  logs_dir: str,
                  syntax_file_path: str,
                  output_file: str,
                  k: int = 5,
                  debug=False,
                  rs: bool = False):
    # helper function
    flatten = lambda l: [x for xs in l for x in xs]

    output_file = open(output_file, "a")
    if debug:
        print("Entering data split phase")

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    model: Graph = PrefixTreeFactory.get_prefix_tree(traces_dir=logs_dir,
                                                     config_file_path=syntax_file_path,
                                                     remove_trivial_loops=False,
                                                     syntax_tree=syntax_tree,
                                                     one_state_per_template=True)

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

    # Get all traces and randomize their order
    traces: List[str] = os.listdir(logs_dir)
    random.shuffle(traces)

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(traces, k)
    time_results = []
    for i in range(k):
        # Current test fold is i
        test_trace_num: int = i

        # Current validation fold is (i + 1) % k
        validation_trace_num: int = (i + 1) % k

        # All other folds are train traces
        train_traces_nums: List[int] = [x for x in range(k)
                                        if (x != test_trace_num and x != validation_trace_num)]

        test_traces = folds[test_trace_num]
        validation_traces = folds[validation_trace_num]
        train_traces = flatten([folds[x] for x in train_traces_nums])

        # For each train trace
        for train_trace in train_traces:
            # Create new name
            old_name = os.path.join(logs_dir, train_trace)
            new_name = os.path.join(train_dir, train_trace)

            # Move the trace
            os.rename(old_name, new_name)

        # For each validation trace
        for validation_trace_name in validation_traces:
            # Create new name
            old_name = os.path.join(logs_dir, validation_trace_name)
            new_name = os.path.join(validation_dir_true, validation_trace_name)

            # Create a false trace
            produce_false_trace(old_name,
                                os.path.join(validation_dir_false, validation_trace_name),
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

        # Minimize the model...
        initial_solution = PrefixTreeFactory. \
            get_prefix_tree(train_dir, cfg_file,
                            remove_trivial_loops=False,
                            one_state_per_template=True,
                            syntax_tree=syntax_tree)
        # PSA
        schedule3 = LundySchedule(alpha=2.546e-1,
                                  neighborhood_size=len(initial_solution.outgoing_edges),
                                  sample_ratio=1 / 10)
        sa = SimulatedAnnealing(initial_solution, s_tree,
                                validation_dir_true,
                                validation_dir_false,
                                TournamentSelection(initial_solution, 5, 0.75),
                                schedule3)

        import time
        start = time.time()
        solutions: List[Graph] = sa.search_mo(front_size=16,
                                              max_interations=10,
                                              debug=debug)

        end = time.time()
        time_results.append(end - start)

        evaluator: Evaluator = Evaluator(initial_solution, s_tree,
                                         test_dir_true, test_dir_false)

        true_trace_trees: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(test_dir_true, s_tree, True)
        false_trace_trees: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(test_dir_false, s_tree, True)

        output_file.write(f"Iteration {i} {len(solutions)} {end - start}\n")

        for solution in solutions:
            evaluator.update(solution)
            e2, sz, sp, re = evaluator.evaluate(true_trees=true_trace_trees,
                                                false_trees=false_trace_trees)
            output_file.write(f"{len(solution)} {e2} {sz} {sp} {re}\n")

        print("FINISHED ITERATION!!!!!!")

        # Clean up directories
        for train_trace in train_traces:
            old_name = os.path.join(logs_dir, train_trace)
            new_name = os.path.join(train_dir, train_trace)
            os.rename(new_name, old_name)
        for validation_trace_name in validation_traces:
            old_name = os.path.join(logs_dir, validation_trace_name)
            new_name = os.path.join(validation_dir_true, validation_trace_name)
            false_trace = os.path.join(validation_dir_false, validation_trace_name)
            os.remove(false_trace)
            os.rename(new_name, old_name)
        for test_trace_name in test_traces:
            old_name = os.path.join(logs_dir, test_trace_name)
            new_name = os.path.join(test_dir_true, test_trace_name)
            false_trace = os.path.join(test_dir_false, test_trace_name)
            os.remove(false_trace)
            os.rename(new_name, old_name)

    os.rmdir(test_dir_true)
    os.rmdir(test_dir_false)
    os.rmdir(validation_dir_true)
    os.rmdir(validation_dir_false)
    os.rmdir(train_dir)
    output_file.close()
    print(np.mean(time_results))


def evaluate_accuracy(syntax_tree: SyntaxTree,
                      logs_dir: str,
                      syntax_file_path: str,
                      k: int = 5,
                      debug=False,
                      rs: bool = False):
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Get all traces in the directory
    traces: List[str] = os.listdir(logs_dir)

    number_of_traces: int = 10

    while number_of_traces <= 30:

        print(f"Starting accuracy test with {number_of_traces} traces.")

        seed: int = number_of_traces
        random.seed(seed + 1)

        # Select the next pool of traces
        current_trace_pool = random.sample(traces, k=number_of_traces)

        # The directory for the selected traces
        new_dir: str = logs_dir + f"_kfcv_{number_of_traces}"
        os.mkdir(new_dir)

        for current_trace in current_trace_pool:
            # Create new name
            old_name = os.path.join(logs_dir, current_trace)
            new_name = os.path.join(new_dir, current_trace)

            # Move the trace
            os.rename(old_name, new_name)

        start = time.time()
        file_to_write: str = f"kfcv_{number_of_traces}_output.txt"
        if rs:
            file_to_write = "prs_" + file_to_write
        # Perform KFCV here
        kfcv_accuracy(
            syntax_tree,
            new_dir,
            syntax_file_path,
            file_to_write,
            k=k,
            debug=debug,
            rs=rs)

        end = time.time()

        print(f"Iteration took {end - start} seconds.")

        # Move the traces back into the original directory
        for current_trace in current_trace_pool:
            # Create new name
            old_name = os.path.join(logs_dir, current_trace)
            new_name = os.path.join(new_dir, current_trace)

            # Move the trace
            os.rename(new_name, old_name)

        print(f"Finished KFCV with {number_of_traces} traces.")

        # Remove the empty directory
        os.rmdir(new_dir)

        number_of_traces += 10


def evaluate_scalability(syntax_tree: SyntaxTree,
                         logs_dir: str,
                         syntax_file_path: str,
                         output_file: str,
                         debug=False):
    # helper function
    flatten = lambda l: [x for xs in l for x in xs]

    # Partition size
    k = 5

    output_file = open(output_file, "a")
    if debug:
        print("Entering data split phase")

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    model: Graph = PrefixTreeFactory.get_prefix_tree(traces_dir=logs_dir,
                                                     config_file_path=syntax_file_path,
                                                     remove_trivial_loops=False,
                                                     syntax_tree=syntax_tree)

    # Create directory names
    train_dir: str = logs_dir + "_train"

    validation_dir: str = logs_dir + "_validation"
    validation_dir_true: str = validation_dir + "_true"
    validation_dir_false: str = validation_dir + "_false"

    os.mkdir(validation_dir_true)
    os.mkdir(validation_dir_false)
    os.mkdir(train_dir)

    # Get all traces and randomize their order
    traces: List[str] = os.listdir(logs_dir)
    random.shuffle(traces)

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(traces, k)

    # Assign first partition to the validation set
    validation_trace_num: int = 0

    # Assign all other partitions to the train set
    train_traces_nums: List[int] = list(range(1, k))

    validation_traces = folds[validation_trace_num]
    train_traces = flatten([folds[x] for x in train_traces_nums])

    # For each train trace
    for train_trace in train_traces:
        # Create new name
        old_name = os.path.join(logs_dir, train_trace)
        new_name = os.path.join(train_dir, train_trace)

        # Move the trace
        os.rename(old_name, new_name)

    # For each validation trace
    for validation_trace_name in validation_traces:
        # Create new name
        old_name = os.path.join(logs_dir, validation_trace_name)
        new_name = os.path.join(validation_dir_true, validation_trace_name)

        # Create a false trace
        produce_false_trace(old_name,
                            os.path.join(validation_dir_false, validation_trace_name),
                            syntax_tree,
                            model)

        # Move the trace
        os.rename(old_name, new_name)

    # Minimize the model...
    initial_solution = PrefixTreeFactory. \
        get_prefix_tree(train_dir, cfg_file,
                        remove_trivial_loops=False,
                        one_state_per_template=True,
                        syntax_tree=syntax_tree)

    schedule3 = LundySchedule(alpha=2.546e-1,
                              neighborhood_size=len(initial_solution.outgoing_edges),
                              sample_ratio=1 / 10)

    sa = SimulatedAnnealing(initial_solution, s_tree,
                            validation_dir_true,
                            validation_dir_false,
                            TournamentSelection(initial_solution, 5, 0.75),
                            schedule3)

    start = time.time()

    solutions: List[Graph] = sa.search_mo(front_size=16,
                                          max_interations=10,
                                          debug=debug)

    end = time.time()

    output_file.write(f"Iteration {len(solutions)} {end - start}\n")

    for solution in solutions:
        output_file.write(f"{len(solution)}\n")

    # Clean up directories
    for train_trace in train_traces:
        old_name = os.path.join(logs_dir, train_trace)
        new_name = os.path.join(train_dir, train_trace)
        os.rename(new_name, old_name)
    for validation_trace_name in validation_traces:
        old_name = os.path.join(logs_dir, validation_trace_name)
        new_name = os.path.join(validation_dir_true, validation_trace_name)
        false_trace = os.path.join(validation_dir_false, validation_trace_name)
        os.remove(false_trace)
        os.rename(new_name, old_name)

    os.rmdir(validation_dir_true)
    os.rmdir(validation_dir_false)
    os.rmdir(train_dir)
    output_file.close()


def evaluate_scalability_all(syntax_tree: SyntaxTree,
                             logs_dir: str,
                             syntax_file_path: str,
                             k: int = 5,
                             debug=False):
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Get all traces in the directory
    traces: List[str] = os.listdir(logs_dir)

    number_of_traces: int = 950

    while number_of_traces <= 1000:

        print(f"Starting KFCV with {number_of_traces} traces.")

        seed: int = number_of_traces + 1
        random.seed(seed)

        # The directory for the selected traces
        new_dir: str = logs_dir + f"_scalability_{number_of_traces}"
        os.mkdir(new_dir)

        i_start = time.time()

        # Perform k tests
        for i in range(k):

            start = time.time()

            # Select the next pool of traces
            current_trace_pool = random.sample(traces, k=number_of_traces)

            for current_trace in current_trace_pool:
                # Create new name
                old_name = os.path.join(logs_dir, current_trace)
                new_name = os.path.join(new_dir, current_trace)

                # Move the trace
                os.rename(old_name, new_name)

            # Perform scalability test here
            evaluate_scalability(
                syntax_tree,
                new_dir,
                syntax_file_path,
                f"sosa_scalability_{number_of_traces}_output.txt",
                debug=debug)

            # Move the traces back into the original directory
            for current_trace in current_trace_pool:
                # Create new name
                old_name = os.path.join(logs_dir, current_trace)
                new_name = os.path.join(new_dir, current_trace)

                # Move the trace
                os.rename(new_name, old_name)

            end = time.time()
            print(f"Iteration {i + 1} out of {k} took {end - start} seconds.")

        os.rmdir(new_dir)

        i_end = time.time()

        print(f"Finished scalability test for {number_of_traces} traces. It took {i_end - i_start}")

        number_of_traces += 50


if __name__ == '__main__':
    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
    cfg_file = str(project_root.joinpath('tests/resources/config.json'))
    all_logs = str(project_root.joinpath('tests/resources/all_traces'))
    s_tree = SyntaxTreeFactory().parse_file(cfg_file)
    output_file = "output_kfold_test2.txt"
    evaluate_accuracy(s_tree,
                      all_logs,
                      cfg_file,
                      k=5,
                      debug=True,
                      rs=True)
    # evaluate_scalability_all(s_tree,
    #                          all_logs,
    #                          cfg_file,
    #                          k=5,
    #                          debug=False)