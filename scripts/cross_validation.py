import os
import shutil
from multiprocessing import Pool

import numpy as np
from numpy import array_split, mean, std
from pathlib import Path
from random import shuffle
import random
from typing import Tuple, List

from scripts.log_scrambler import produce_false_trace
from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.reinforcementlearning.reinforcement_learning import \
    ReinforcementLearning
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory

random.seed(os.environ['random_seed'] if 'random_seed' in os.environ else 5)

DEFAULT_ALPHA = 0.2
DEFAULT_GAMMA = 0.7
DEFAULT_EPSILON = 0.3
DEFAULT_PICKLE_PATH = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle")

random.seed(0)


def k_fold_cross_validation(k: int,
                            dataset_size: int,
                            logs_dir: str,
                            k_fold_data_path: Path = PROJECT_ROOT.joinpath("resources/k_fold_traces"),
                            config_location: str = PROJECT_ROOT.joinpath("resources/config.json"),
                            debug=False):
    """
    Performs k-fold cross validation on a given set of traces, collecting recall and specificity.
    :param syntax_tree: The syntax tree used to match a particular log entry.
    :param logs_dir: The directory containing the traces.
    :param k: The number of folds.
    :param config_location: The location of the configuration used to build the syntax tree.
    :param debug: Whether or not to print debug information to the console.
    """
    # Check if log directory exists
    logs = os.listdir(logs_dir)
    logs = [Path(logs_dir).joinpath(line) for line in logs]
    # print(logs)
    for file in os.listdir(k_fold_data_path):
        shutil.rmtree(k_fold_data_path.joinpath(file))

    # Get all traces and randomize their order
    traces = []
    os.mkdir(k_fold_data_path.joinpath("traces"))
    # Create traces
    for i in range(dataset_size):
        rand = random.randint(0, len(logs) - 1)
        for j in range(k):

            shutil.copy(logs[rand],
                        k_fold_data_path.joinpath(f"traces/xx{i}"))
        traces.append(logs[rand])
        logs.remove(logs[rand])

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(traces, k)

    if debug:
        print(f"Entering k-fold cross validation, with {len(os.listdir(k_fold_data_path.joinpath('traces')))} "
              f"traces split into {len(folds)} folds...")

    alpha = DEFAULT_ALPHA
    gamma = DEFAULT_GAMMA
    epsilon = DEFAULT_EPSILON

    st = SyntaxTreeFactory().parse_file(config_location)

    pool_size = k
    seeds = list(range(k))
    print(f"Seeds selected: {seeds}")

    print(f"Creating dataset for each fold...")
    with Pool(pool_size) as p:
        p.starmap(create_data_set, zip(seeds, [traces] * k, folds))

    # for index, fold in folds:
    prefix_tree_pickle_paths = []
    for i in range(k):
        prefix_tree_pickle_paths.append(PROJECT_ROOT.joinpath(f"resources/k_fold_traces/{i}/prefix_tree.pickle"))

    print(f"Entering parallel approach execution for each fold...")
    with Pool(pool_size) as p:
        p.starmap(run_algorithm, zip(seeds, [alpha] * len(seeds),
                                     [epsilon] * len(seeds),
                                     [gamma] * len(seeds),
                                     [st] * len(seeds),
                                     prefix_tree_pickle_paths,
                                     ))
    print("Collecting metrics..")
    collect_metrics(k)


def create_data_set(seed, traces, fold):
    DatasetFactory(PROJECT_ROOT.joinpath("resources/k_fold_traces")).create_data_set_with_fold(traces, fold, seed)


def run_algorithm(seed: int, alpha: float, epsilon: float, gamma: float, st: SyntaxTree, prefix_tree_pickle_path: str):
    random.seed(seed)
    positive_traces, negative_traces = DatasetFactory(PROJECT_ROOT.joinpath(f"resources/k_fold_traces/{seed}")).get_evaluation_traces(st, (PROJECT_ROOT.joinpath(f"resources/k_fold_traces/{seed}")))

    rl = ReinforcementLearning(alpha, gamma, epsilon, positive_traces, negative_traces, st, prefix_tree_pickle_path)

    # atexit.register(exit_handler, rl)

    rl.run(id=seed, debug=False)

    print(f"Seed {seed} completed!")
    exit_handler(rl, seed=seed)


def collect_metrics(k: int):
    metrics = {"best": None, "worst": None, "middle": None}
    for m in metrics:
        metrics[m] = np.zeros(11)
    metrics["initial"] = np.zeros(7)

    for i in range(k):
        with open(PROJECT_ROOT.joinpath(f"out/metrics_{i}.csv"), 'r') as f:
            lines = f.readlines()[1:]
            lines = [l.strip() for l in lines]
            lines = [l.split(", ") for l in lines]
            lines = [[float(item) for item in l] for l in lines]

            best_index = 0
            worst_index = 0
            mid_index = 0
            max_steps = -1
            min_steps = 10000

            for index, line in enumerate(lines):
                if line[-1] > max_steps:
                    max_steps = line[-1]
                    best_index = index
                if line[-1] < min_steps:
                    min_steps = line[-1]
                    worst_index = index
            av = (max_steps + min_steps) / 2
            for index, line in enumerate(lines):
                if av - 1 <= line[-1] <= av + 1:
                    mid_index = index
                    break
            metrics["best"] += lines[best_index]
            metrics["worst"] += lines[worst_index]
            metrics["middle"] += lines[mid_index]
        pt = PrefixTreeFactory().unpickle_tree(PROJECT_ROOT.joinpath(f"resources/k_fold_traces/{i}/prefix_tree.pickle"))
        st = SyntaxTreeFactory().parse_file(PROJECT_ROOT.joinpath("resources/config.json"))
        positive_traces, negative_traces = DatasetFactory(PROJECT_ROOT.joinpath(f"resources/k_fold_traces/{i}")).get_evaluation_traces(st, PROJECT_ROOT.joinpath(f"resources/k_fold_traces/{i}"))
        ev = Evaluator(pt, st, positive_traces, negative_traces)
        precision, recall, f1score, specificity, size = ev.evaluate()
        nodes = pt.get_number_of_nodes()
        transitions = pt.get_number_of_transitions()
        metrics["initial"] += [f1score, recall, specificity, precision, size, nodes, transitions]

    for m in metrics:
        metrics[m] /= k

    with open(PROJECT_ROOT.joinpath("out/metrics.csv"), 'w+') as f:
        for m in metrics:
            f.write(m + " & ")
            [f.write("{:.2f}".format(x) + " & ") for x in metrics[m]]
            f.write("\n")
    print(f"Metrics have been save at {PROJECT_ROOT.joinpath('out/metrics.csv')}")


def exit_handler(rl: ReinforcementLearning, seed=-1):
    print("EXITING!")

    with open(PROJECT_ROOT.joinpath(f"out/metrics_{seed}.csv"), 'w+') as f:
        f.write("episode, reward, f1_score, recall, specificity, precision, size, nodes, transitions, duration, steps\n")
        for i in range(rl.episodes):
            f.write(str(i)
                    + ", " + str(rl.total_rewards[i])
                    + ", " + str(rl.total_f1scores[i])
                    + ", " + str(rl.total_recalls[i])
                    + ", " + str(rl.total_specificities[i])
                    + ", " + str(rl.total_precisions[i])
                    + ", " + str(rl.total_sizes[i])
                    + ", " + str(rl.total_nodes[i])
                    + ", " + str(rl.total_transitions[i])
                    + ", " + str(rl.total_duration[i])
                    + ", " + str(rl.total_steps[i])
                    + "\n")


# Example usage
if __name__ == '__main__':
    data_sets = [200, 1000]
    for size in data_sets:
        data_path = PROJECT_ROOT.joinpath(f"resources/k_fold_metrics/{size}")
        if os.path.exists(data_path):
            for file in os.listdir(data_path):
                shutil.rmtree(data_path.joinpath(file))
        else:
            os.mkdir(data_path)

        cfg_file = str(PROJECT_ROOT.joinpath('resources/config.json'))
        if size != 400:
            k_fold_cross_validation(5, size, PROJECT_ROOT.joinpath("resources/traces_large"), debug=True)

        print("Copying files..")
        shutil.copy(PROJECT_ROOT.joinpath("out/metrics.csv"), data_path.joinpath("metrics.csv"))
        shutil.copy(PROJECT_ROOT.joinpath("out/metrics_0.csv"), data_path.joinpath("metrics_0.csv"))
        shutil.copy(PROJECT_ROOT.joinpath("out/metrics_1.csv"), data_path.joinpath("metrics_1.csv"))
        shutil.copy(PROJECT_ROOT.joinpath("out/metrics_2.csv"), data_path.joinpath("metrics_2.csv"))
        shutil.copy(PROJECT_ROOT.joinpath("out/metrics_3.csv"), data_path.joinpath("metrics_3.csv"))
        shutil.copy(PROJECT_ROOT.joinpath("out/metrics_4.csv"), data_path.joinpath("metrics_4.csv"))
