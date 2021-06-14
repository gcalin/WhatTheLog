from datetime import timedelta
from time import time

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory

import copy


def main():
    st = SyntaxTreeFactory().parse_file(
        PROJECT_ROOT.joinpath("resources/config.json"))

    print("Reading positive and negative traces..")
    positive_traces, negative_traces = DatasetFactory.get_evaluation_traces(st, PROJECT_ROOT.joinpath("resources/data"))
    print("Finished reading positive and negative traces..")

    pt = PrefixTreeFactory().unpickle_tree(PROJECT_ROOT.joinpath("resources/prefix_tree.pickle"))

    # new_tree = copy.deepcopy(pt)

    ev = Evaluator(pt, st, positive_traces, negative_traces)

    print(ev.evaluate())
    #
    start = pt.start_node

    for _ in range(40):
        first = pt.get_outgoing_states(start)[0]
        pt.full_merge_states_with_children(first, set())

    #
    print(ev.evaluate())

    # print(pt.get_outgoing_states(first))
    Visualizer(pt).visualize("finaltree.png")


def dfs(pt: Graph):
    st = SyntaxTreeFactory().parse_file(
        PROJECT_ROOT.joinpath("resources/config.json"))

    print("Reading positive and negative traces..")
    positive_traces, negative_traces = DatasetFactory.get_evaluation_traces(st, PROJECT_ROOT.joinpath("resources/data"))
    print("Finished reading positive and negative traces..")

    start = time()
    stack = []
    stack.append(pt.start_node)
    visited = set()
    nodes = 0
    ev = Evaluator(pt, st, positive_traces, negative_traces)

    while len(stack) > 0:
        node = stack.pop()
        node = pt.full_merge_states_with_children(node)
        ev.evaluate(0.5, 0.5)
        nodes += 1
        visited.add(node)
        stack += [out for out in pt.get_outgoing_states(node) if out not in visited]
        stack.append(node)
        stack = list(filter(lambda x: x in pt and x not in visited, stack))
    print(nodes)
    print(pt.size())
    print(f"Finished! Time elapsed: {timedelta(seconds=time() - start)}")


if __name__ == '__main__':
    main()