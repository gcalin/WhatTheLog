import pandas as pd

import whatthelog.reinforcementlearning.environment
from whatthelog.reinforcementlearning.actionspace import ActionSpace, Actions
from whatthelog.reinforcementlearning.environment import GraphEnv
from whatthelog.definitions import PREFIX_TREE_PICKLE_PATH, CONFIG_FILE
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


def main():
    st = SyntaxTreeFactory().parse_file(CONFIG_FILE)
    temp_env = GraphEnv(PREFIX_TREE_PICKLE_PATH, st, None, None)
    action_space = ActionSpace(Actions)
    valid_actions = pd.DataFrame(columns=action_space.actions, index=range(temp_env.observation_space.n))

    decoder = {v: k for k, v in temp_env.state_mapping.items()}
    new_index = []

    for state in range(temp_env.observation_space.n):
        value = decoder[state]
        edges = int(value[0])
        entropy = int(value[1])
        new_index.append(f"{state} ({edges}.{entropy})")
        actions = action_space.get_valid_actions(edges, entropy)
        new_row = list(map(lambda x: int(x in actions), action_space.actions))

        valid_actions.iloc[state] = pd.Series(new_row)

    valid_actions.index = new_index
    print(valid_actions.to_latex())


if __name__ == '__main__':
    main()