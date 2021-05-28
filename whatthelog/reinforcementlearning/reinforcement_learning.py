import random
import time
import numpy as np
from numpy.compat import long

from scripts.match_trace import match_trace
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.reinforcementlearning.environment import GraphEnv
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


if __name__ == '__main__':
    a = long(time.time() * 256)
    random.seed(a)
    st = SyntaxTreeFactory().parse_file(
        PROJECT_ROOT.joinpath("resources/config.json"))
    pt: PrefixTree = PrefixTreeFactory().get_prefix_tree(
        PROJECT_ROOT.joinpath("resources/traces"),
        PROJECT_ROOT.joinpath("resources/config.json"))

    PrefixTreeFactory().pickle_tree(pt, PROJECT_ROOT.joinpath(
        "out/prefixtree.pickle"))

    evaluator = Evaluator(pt, st,
                          PROJECT_ROOT.joinpath("resources/traces"),
                          PROJECT_ROOT.joinpath(
                              "resources/negative_traces"))

    env = GraphEnv(pt, st)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyper-parameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    epochs = 1
    epoch = 0

    env.seed()
    for i in range(epochs):
        state = env.reset()
        total_reward = 0

        done = False

        while not done:
            actions = env.get_valid_actions()

            if random.random() < epsilon:
                index = random.randint(0, len(actions) - 1)
                action = actions[index]
            else:
                random.shuffle(actions)
                action = max(list(enumerate(q_table[state, actions])),
                             key=lambda x: x[1])[0]

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            old_value = q_table[state][action]
            max_q = np.max(q_table[next_state])
            q_table[state][action] = (1 - alpha) * old_value + alpha * (
                    reward + gamma * max_q)
            state = next_state
            epoch += 1

        print(f"Epoch {i} completed with reward: {total_reward}!")
        print("q table: \n", q_table)
        Visualizer(env.graph).visualize(f"{i} epoch.png")

    print("Timesteps taken: {}".format(epochs))
