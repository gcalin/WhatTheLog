import os
import random
import sys
from datetime import timedelta
from pathlib import Path
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

from whatthelog.syntaxtree.syntax_tree import SyntaxTree


sys.path.append(Path(os.path.abspath(__file__)).parent.parent.parent.__str__())

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.reinforcementlearning.environment import GraphEnv
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


class ReinforcementLearning:

    def __init__(self, alpha: float, gamma: float, epsilon: float,
                 positive_traces: List[List[str]], negative_traces: List[List[str]],
                 syntax_tree: SyntaxTree, prefix_tree_pickle_path: str = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle"),):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = 0

        print(f"Creating environment with perfix tree: {prefix_tree_pickle_path}")
        self.env = GraphEnv(
            prefix_tree_pickle_path, syntax_tree,
            positive_traces,
            negative_traces)

        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        precision, recall, f1score, specificity, size = self.env.evaluator.evaluate()
        print(precision, recall, f1score, specificity, size, self.env.graph.get_number_of_nodes(), self.env.graph.get_number_of_transitions())

        self.total_rewards = []
        self.total_f1scores = []
        self.total_recalls = []
        self.total_specificities = []
        self.total_precisions = []
        self.total_sizes = []
        self.total_nodes = []
        self.total_transitions = []
        self.total_duration = []
        self.total_steps = []

        self.min_epsilon = 0.1

    def run(self, id: int=-1, debug: bool=False):
        with open(PROJECT_ROOT.joinpath("out/parameters.csv"), "w+") as f:
            f.write("alpha, epsilon, gamma\n")
            f.write(str(self.alpha) + ", " + str(self.epsilon) + ", " + str(self.gamma))
        stop = False
        prev = 0
        count = 0
        steps = 0

        while not stop and self.episodes < 3000:
            start_time = time()

            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            rewards = []
            if self.epsilon > self.min_epsilon:
                self.epsilon -= 0.003
            while not done and not stop:
                actions = self.env.get_valid_actions()
                if random.random() < self.epsilon:
                    index = random.randint(0, len(actions) - 1)
                    action = actions[index]
                else:

                    random.shuffle(actions)
                    action = max(list(zip(actions, self.q_table[state, actions])),
                                 key=lambda x: x[1])[0]
                # print(action, self.epsilon, actions)

                next_state, reward, done, info = self.env.step(action)
                stop = info["stop"]
                # print(stop, reward)
                # print(next_state, reward, done, info)

                total_reward += reward

                old_value = self.q_table[state][action]
                max_q = np.max(self.q_table[next_state])
                self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * (
                        reward + self.gamma * max_q)
                state = next_state
                steps += 1
                rewards.append(total_reward)

            if np.abs(steps - prev) <= 1:
                count += 1
            else:
                count = 0
            if count >= 10:
                stop = True

            prev = steps

            duration = timedelta(seconds=time() - start_time).total_seconds()
            # pandas.DataFrame(self.q_table).to_csv(PROJECT_ROOT.joinpath(f"out/q_values/q_values_{self.episodes}"))
            self.episodes += 1
            precision, recall, f1score, specificity, size = self.env.evaluator.evaluate()

            self.total_nodes.append(self.env.graph.get_number_of_nodes())
            self.total_transitions.append(self.env.graph.get_number_of_transitions())

            self.total_rewards.append(total_reward)
            self.total_f1scores.append(f1score)
            self.total_recalls.append(recall)
            self.total_specificities.append(specificity)
            self.total_precisions.append(precision)
            self.total_sizes.append(size)
            self.total_duration.append(duration)
            self.total_steps.append(steps)

            if debug:
                print(f"Episode {self.episodes} completed with total steps: {self.env.step_count}.")
