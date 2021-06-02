import random
from time import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.reinforcementlearning.environment import GraphEnv
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
from whatthelog.datasetcreator.dataset_factory import DatasetFactory


if __name__ == '__main__':
    random.seed(5)
    st = SyntaxTreeFactory().parse_file(
        PROJECT_ROOT.joinpath("resources/config.json"))

    print("Reading positive and negative traces..")
    positive_traces, negative_traces = DatasetFactory.get_evaluation_traces(st, PROJECT_ROOT.joinpath("resources/data"))
    print("Finished reading positive and negative traces..")

    # pt: PrefixTree = PrefixTreeFactory().get_prefix_tree(
    #     PROJECT_ROOT.joinpath("resources/data/traces"),
    #     PROJECT_ROOT.joinpath("resources/config.json"), remove_trivial_loops=True)
    #
    # PrefixTreeFactory().pickle_tree(pt, PROJECT_ROOT.joinpath("resources/prefix_tree_trivial_loops.pickle"))

    env = GraphEnv(PROJECT_ROOT.joinpath("resources/prefix_tree.pickle"), st,
                   positive_traces,
                   negative_traces)

    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    # Hyper-parameters
    alpha = 0.1
    gamma = 1
    epsilon = 0.1
    epochs = 20
    step = 1

    total_rewards = []

    follow_custom_policy = False

    print(env.graph.size())

    x = [x in env.graph for x in env.graph.states.values()]

    for i in range(epochs):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:

            start_time = time()
            if follow_custom_policy:
                action = policy[step]
            else:
                actions = env.get_valid_actions()
                if step == 2:
                    print(actions)
                if random.random() < epsilon:
                    index = random.randint(0, len(actions) - 1)
                    action = actions[index]
                else:

                    random.shuffle(actions)
                    action = max(list(zip(actions, q_table[state, actions])),
                                 key=lambda x: x[1])[0]

            next_state, reward, done, info = env.step(action)
            # print(done)
            total_reward += reward

            # print(f"state: {state}, action: {action}, reward: {reward}")

            old_value = q_table[state][action]
            max_q = np.max(q_table[next_state])
            q_table[state][action] = (1 - alpha) * old_value + alpha * (
                    reward + gamma * max_q)
            state = next_state
            # Visualizer(env.graph).visualize(f"run/{step}.png")
            # print(env.action_sequence)
            step += 1
        total_rewards.append(total_reward)
        print(env.action_sequence)
        # print(q_table)
        print(f"Epoch {i} completed with total reward: {total_reward}.")

    Visualizer(env.graph).visualize("final.png")
    fitness = env.evaluator.evaluate(0.5, 0.5)
    print("fitness: ", fitness)

    print(f"total_reward: {sum(total_rewards)}")
    print(q_table)
    plt.rc('axes', labelsize=15)
    plt.plot(list(range(epochs)), total_rewards)
    plt.ylabel("Total reward", labelpad=15)
    plt.xlabel("Epoch", labelpad=15)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PROJECT_ROOT.joinpath("out/plots/small tree 1000 epochs rewards.png"))
    plt.show()

    pd.DataFrame(q_table).to_csv(PROJECT_ROOT.joinpath("out/q_table.csv"))
