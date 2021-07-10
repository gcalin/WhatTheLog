import atexit
import os
import sys
import random
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import sys
import atexit

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.reinforcementlearning.reinforcement_learning import \
    ReinforcementLearning
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory

DEFAULT_ALPHA = 0.2
DEFAULT_GAMMA = 0.7
DEFAULT_EPSILON = 0.3
DEFAULT_EPISODES = 214
DEFAULT_PICKLE_PATH = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle")


def main(arguments):

    alpha = arguments[0] if len(arguments) > 0 else DEFAULT_ALPHA
    gamma = arguments[1] if len(arguments) > 1 else DEFAULT_GAMMA
    epsilon = arguments[2] if len(arguments) > 2 else DEFAULT_EPSILON
    episodes = arguments[3] if len(arguments) > 3 else DEFAULT_EPISODES
    prefix_tree_pickle_path = arguments[4] if len(arguments) > 4 else DEFAULT_PICKLE_PATH

    st = SyntaxTreeFactory().parse_file(PROJECT_ROOT.joinpath("resources/config.json"))
    positive_traces, negative_traces = DatasetFactory(PROJECT_ROOT.joinpath("resources/data/traces")).get_evaluation_traces(st, PROJECT_ROOT.joinpath("resources/data/"))

    pool_size = 5
    seeds = [0, 1, 2, 3, 4]
    #
    with Pool(pool_size) as p:
        p.starmap(run_algorithm, zip(seeds, [alpha] * len(seeds),
                                     [episodes] * len(seeds),
                                     [epsilon] * len(seeds),
                                     [gamma] * len(seeds),
                                     [st] * len(seeds),
                                     [prefix_tree_pickle_path] * len(seeds),
                                     [positive_traces] * len(seeds),
                                     [negative_traces] * len(seeds)
                                     ))
    # run_algorithm(seeds[0], alpha, episodes, epsilon, gamma, st, prefix_tree_pickle_path, positive_traces, negative_traces)
    # combine_results()


def run_algorithm(seed: int, alpha: float, episodes: int, epsilon: float, gamma: float, st: SyntaxTree, prefix_tree_pickle_path: str,
                  positive_traces: List[List[str]],
                  negative_traces: List[List[str]]):
    random.seed(seed)

    rl = ReinforcementLearning(alpha, gamma, epsilon, episodes, positive_traces, negative_traces, st, prefix_tree_pickle_path)

    # atexit.register(exit_handler, rl)

    rewards, f1scores, recalls, specificities, precisions, sizes, nodes, transitions = rl.run(return_metrics=True, id=seed)

    print(f"Seed {seed} completed!")
    exit_handler(rl, seed=seed)
    # with open(PROJECT_ROOT.joinpath(f"out/metrics_{seed}_ended.txt"), 'w+') as file:
    #     [file.write(str(v) + ", ") for v in rewards]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in f1scores]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in recalls]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in specificities]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in precisions]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in sizes]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in nodes]
    #     file.write("\n")
    #     [file.write(str(v) + ", ") for v in transitions]
    #     file.write("\n")

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

def combine_results():
    """
    Method that combines individual results into one file.

    :param output_name:
    :return:
    """
    total_rewards = np.zeros(DEFAULT_EPISODES)
    total_sizes = np.zeros(DEFAULT_EPISODES)
    total_f1scores = np.zeros(DEFAULT_EPISODES)
    total_recalls = np.zeros(DEFAULT_EPISODES)
    total_precisions = np.zeros(DEFAULT_EPISODES)
    total_specificities = np.zeros(DEFAULT_EPISODES)
    total_nodes = np.zeros(DEFAULT_EPISODES)
    total_transitions = np.zeros(DEFAULT_EPISODES)
    duration = 0

    for i in range(5):
        with open(PROJECT_ROOT.joinpath(f"out/metrics_{i}.txt"), 'r') as f:
            lines = f.readlines()
            lines = [line[:-3] for line in lines]
            lines = [line.split(", ") for line in lines]
            lines = [[float(value) for value in line] for line in lines]
            total_rewards += lines[0]
            total_f1scores += lines[1]
            total_recalls += lines[2]
            total_specificities += lines[3]
            total_precisions += lines[4]
            total_sizes += lines[5]
            total_nodes += lines[6]
            total_transitions += lines[7]

        with open(PROJECT_ROOT.joinpath(f"out/duration_{i}.txt"), 'r') as f:
            duration += float(f.read())

    total_rewards /= 5
    total_sizes /= 5
    total_f1scores /= 5
    total_recalls /= 5
    total_precisions /= 5
    total_specificities /= 5
    total_nodes /= 5
    total_transitions /= 5
    duration /= 5

    with open(PROJECT_ROOT.joinpath("out/metrics.csv"), 'w+') as f:
        f.write("episode, reward, f1_score, recall, specificity, precision, size, nodes, transitions\n")
        for i in range(len(total_rewards)):
            f.write(str(i)
                    + ", " + str(total_rewards[i])
                    + ", " + str(total_f1scores[i])
                    + ", " + str(total_recalls[i])
                    + ", " + str(total_specificities[i])
                    + ", " + str(total_precisions[i])
                    + ", " + str(total_sizes[i])
                    + ", " + str(total_nodes[i])
                    + ", " + str(total_transitions[i])
                    + "\n")

    with open(PROJECT_ROOT.joinpath("out/duration.csv"), 'w+') as f:
        f.write("duration\n")
        f.write(str(duration))


if __name__ == '__main__':
    main(sys.argv[1:])
