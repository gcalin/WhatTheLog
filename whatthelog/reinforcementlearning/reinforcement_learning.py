import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.reinforcementlearning.environment import GraphEnv
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


if __name__ == '__main__':
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    st = SyntaxTreeFactory().parse_file(
        PROJECT_ROOT.joinpath("resources/config.json"))

    print("Reading positive and negative traces..")
    positive_traces, negative_traces = DatasetFactory.get_evaluation_traces(st,
                                                                            PROJECT_ROOT.joinpath(
                                                                                "resources/data"))
    print("Finished reading positive and negative traces..")

    # Hyper-parameters
    alpha = 0.05
    gamma = 0.5
    epsilon = 0.1
    epochs = 40

    w_accuracy = 0.5
    w_size = 0.5

    env = GraphEnv(PROJECT_ROOT.joinpath("resources/prefix_tree.pickle"), st,
                   positive_traces,
                   negative_traces,
                   w_accuracy, w_size)

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    total_fitnesses = []
    total_rewards = []
    best_fitness = 0

    x = [x in env.graph for x in env.graph.states.values()]

    for i in range(epochs):
        state = env.reset()
        total_reward = 0
        done = False
        if i == 1:
            print("it 1")
        while not done:
            actions = env.get_valid_actions()

            if random.random() < epsilon:
                index = random.randint(0, len(actions) - 1)
                action = actions[index]
            else:

                random.shuffle(actions)
                action = max(list(zip(actions, q_table[state, actions])),
                             key=lambda x: x[1])[0]

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            old_value = q_table[state][action]
            max_q = np.max(q_table[next_state])
            q_table[state][action] = (1 - alpha) * old_value + alpha * (
                    reward + gamma * max_q)
            state = next_state

        fitness = env.evaluator.evaluate(w_accuracy, w_size)
        total_rewards.append(total_reward)
        total_fitnesses.append(fitness)
        if fitness > best_fitness:
            PrefixTreeFactory().pickle_tree(env.graph, PROJECT_ROOT.joinpath(
                f"finaltree.pickle"))
            print(f"saved: fitness: {fitness}")
            best_fitness = fitness

        print(pd.DataFrame(q_table))
        print(f"Epoch {i} completed with total reward: {total_reward}.")

    pt = PrefixTreeFactory().unpickle_tree(PROJECT_ROOT.joinpath("finaltree.pickle"))
    Visualizer(pt).visualize("finaltree.png")

    plt.rc('axes', labelsize=15)
    plt.plot(list(range(epochs)), total_rewards)
    plt.ylabel("Total reward", labelpad=15)
    plt.xlabel("Epoch", labelpad=15)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PROJECT_ROOT.joinpath("out/plots/small tree 1000 epochs rewards.png"))
    plt.show()

    plt.rc('axes', labelsize=15)
    plt.plot(list(range(epochs)), total_fitnesses)
    plt.ylabel("Total reward", labelpad=15)
    plt.xlabel("Epoch", labelpad=15)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PROJECT_ROOT.joinpath("out/plots/small tree 1000 epochs fitnesses.png"))
    plt.show()

    pd.DataFrame(q_table).to_csv(PROJECT_ROOT.joinpath("out/q_table.csv"))
