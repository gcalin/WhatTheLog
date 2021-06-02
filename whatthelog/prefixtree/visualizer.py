from pprint import pprint
from typing import Tuple, Dict, List

import networkx as nx
from pygraphviz import AGraph

from whatthelog.prefixtree.graph import Graph
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.auto_printer import AutoPrinter
from tqdm import tqdm


def print(msg): AutoPrinter.static_print(msg)


class Visualizer(AutoPrinter):
    """
    Class to visualize Prefix Tree.
    """

    def __init__(self, graph: Graph):
        """
        Visualizer constructor.
        :param graph: Graph to visualize
        """
        self.graph = graph
        self.G = nx.DiGraph()
        self.label_mapping = {"": 0}

    def visualize(self, file_name="prefixtree.png"):
        """
        Method to visualize the prefix tree.
        :param file_path: Path to save the file, if None dont save
        :return: None
        """
        print("Visualizing tree...")
        labels, branches, depth = self.__populate_graph()
        A = nx.drawing.nx_agraph.to_agraph(self.G)

        for node in A.nodes():
            node.attr.update(label=labels[int(node.name)])

        A.graph_attr.update(nodesep="0.5")
        A.graph_attr.update(pad="1")
        A.layout('dot')
        A.draw(str(PROJECT_ROOT.joinpath("out/" + file_name)))
        print("Visualization has been saved at: " + str(PROJECT_ROOT.joinpath("out/" + file_name)))

    def __populate_graph(self) -> Tuple[Dict[int, str], int, int]:
        """
        Method that populates the graph by traversing the prefix tree
         using breadth first.
        While traversing also keep track of number of branches
         and maximum depth.
         !WORKS ONLY FOR GRAPHS WITH START NODES CURRENTLY
        :return: 3Tuple containing
                a dictionary mapping unique node ids to labels (log ids),
                number of branches,
                maximum depth of tree
        """

        labels = {id(self.graph.start_node): self.get_label(self.graph.start_node.properties.log_templates)}

        queue = self.graph.get_outgoing_states(self.graph.start_node)
        branches = 1
        depth = 1
        visited = {self.graph.start_node}
        while len(queue) != 0:
            level_size = len(queue)
            while level_size > 0 and len(queue) != 0:
                state = queue.pop(0)
                if state not in visited:
                    for parent in self.graph.get_incoming_states(state):

                        self.G.add_edge(id(parent),
                                    id(state), label=self.graph.outgoing_edges[parent][state])
                    visited.add(state)
                    labels[id(state)] = self.get_label(state.properties.log_templates)

                    children = self.graph.get_outgoing_states(state)

                    if len(children) > 1:
                        branches += len(children) - 1

                    queue += children
                    level_size -= 1
            depth += 1

        return labels, branches, depth

    def get_label(self, log_templates: List[str]) -> str:
        label = ""
        if len(log_templates) > 1:
            label += "["
        for log_template in log_templates:
            if log_template not in self.label_mapping:
                self.label_mapping[log_template] = log_template

            label += str(self.label_mapping[log_template]) + ", "
        label = label[:-2]
        if len(log_templates) > 1:
            label += "]"

        return label
