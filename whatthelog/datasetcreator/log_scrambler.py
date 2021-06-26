import random
from typing import List

import scripts.log_scrambler
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class LogScrambler:
    def __init__(self, prefix_tree: PrefixTree, syntax_tree: SyntaxTree):
        self.prefix_tree = prefix_tree
        self.syntax_tree = syntax_tree

    def get_negative_traces(self, amount: int, traces: List[str],
                            write_to_file=False) -> None:
        """

        :param amount: Amount of negative traces
        :param traces: List of paths of correct traces
        :return: The negative traces
        """

        for i in range(amount):
            rand = random.randint(0, len(traces) - 1)

            output_file = PROJECT_ROOT.joinpath(
                f"resources/data/negative_traces/negative_xx{i}")

            scripts.log_scrambler.produce_false_trace(traces[rand],
                                                      output_file,
                                                      self.syntax_tree,
                                                      self.prefix_tree)
            traces.remove(traces[rand])
