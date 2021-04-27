from typing import Tuple, Dict

import networkx as nx
import matplotlib.pyplot as plt

from whatthelog.prefixtree.prefix_tree import PrefixTree
from networkx.drawing.nx_pydot import graphviz_layout


class Visualizer:
    """
    Class to visualize Prefix Tree.
    """

    def __init__(self, prefix_tree: PrefixTree):
        """
        Visualizer constructor.

        :param prefix_tree: Prefix tree to visualize
        """
        self.prefix_tree = prefix_tree
        self.G = nx.DiGraph()

    def visualize(self, file_path="../resources/prefixtree.png"):
        """
        Method to visualize the prefix tree.

        :param file_path: Path to save the file, if None dont save
        :return: None
        """
        labels, branches, depth = self.__populate_graph()

        plt.figure(1, figsize=(branches + 1, depth / 2 + 1))

        pos = graphviz_layout(self.G, prog="dot")
        nx.draw_networkx_labels(self.G, pos, labels)
        nx.draw(self.G, pos, node_size=500, font_size=6)

        plt.tight_layout()
        if file_path is not None:
            plt.savefig(file_path)

        plt.show()

    def __populate_graph(self) -> Tuple[Dict[int, str], int, int]:
        """
        Method that populates the graph by traversing the prefix tree
         using breadth first.
        While traversing also keep track of number of branches
         and maximum depth.

        :return: 3Tuple containing
                a dictionary mapping unique node ids to labels (log ids),
                number of branches,
                maximum depth of tree
        """
        labels = {id(self.prefix_tree.state): str(self.prefix_tree.state)}

        queue = self.prefix_tree.get_children()
        branches = 1
        depth = 1

        while len(queue) != 0:
            level_size = len(queue)

            while level_size > 0:
                prefix_tree = queue.pop(0)

                self.G.add_edge(id(prefix_tree.get_parent().state),
                                id(prefix_tree.state))

                labels[id(prefix_tree.state)] = str(prefix_tree.state)

                children = prefix_tree.get_children()

                if len(children) > 1:
                    branches += len(children) - 1

                queue += children
                level_size -= 1
            depth += 1

        return labels, branches, depth
