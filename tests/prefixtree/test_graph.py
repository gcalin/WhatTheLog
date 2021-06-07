from copy import deepcopy
from typing import List

from whatthelog.exceptions import StateDoesNotExistException
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.graph import Graph
import pytest

from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State


@pytest.fixture
def graph():
    state0 = State(["0"])
    state1 = State(["1"])
    state2 = State(["2"])
    state3 = State(["3"])
    state4 = State(["4"])

    graph = Graph(state0)

    graph.add_state(state0)
    graph.add_state(state1)
    graph.add_state(state2)
    graph.add_state(state3)
    graph.add_state(state4)

    graph.add_edge(state0, state1, EdgeProperties())
    graph.add_edge(state0, state2, EdgeProperties())
    graph.add_edge(state0, state4, EdgeProperties())
    graph.add_edge(state1, state3, EdgeProperties())

    return graph


@pytest.fixture
def fully_connected_graph():
    # A fully connected 4 state graph.
    states: List[State] = [State(["0"]), State(["1"]), State(["2"]), State(["3"])]

    graph: Graph = Graph()

    for state in states:
        graph.add_state(state)

    for state in states:
        for other_state in states:
            graph.add_edge(state, other_state, EdgeProperties())

    return graph


@pytest.fixture
def graph_2():
    s0 = State(["0"])
    s1 = State(["1"])
    s2 = deepcopy(s0)
    s3 = State(["2"])
    s4 = State(log_templates=[["0"], ["1"]])

    states: List[State] = [s0, s1, s2, s3, s4]

    graph: Graph = Graph()

    for state in states:
        graph.add_state(state)

    graph.add_edge(s0, s0)
    graph.add_edge(s0, s1)
    graph.add_edge(s1, s2)
    graph.add_edge(s0, s3)
    graph.add_edge(s3, s4)

    return graph


@pytest.fixture
def graph_match_1():
    start_state = State([""])
    terminal_state = State(["terminal"], is_terminal=True)
    s0 = State(["0"])
    s1 = State(["1"])
    s2 = State(log_templates=[["3", "4", "5"], ["6", "7"], ["8"]])
    s3 = State(log_templates=[["10"], ["11"]])
    s4 = State(["9"])

    states: List[State] = [start_state, s0, s1, s2, s3, s4, terminal_state]

    graph: Graph = Graph(start_node=None)

    for state in states:
        graph.add_state(state)
    graph.start_node = start_state

    graph.add_edge(start_state, s0)
    graph.add_edge(s0, s0)
    graph.add_edge(s0, s1)
    graph.add_edge(s1, s2)
    graph.add_edge(s2, s3)
    graph.add_edge(s3, s4)
    graph.add_edge(s4, terminal_state)

    return graph


def test_match_tree_1(graph_match_1: Graph):
    start_state = State([""])
    terminal_state = State([""], is_terminal=True)
    pTree = PrefixTree(start_state)
    s0 = State(["0"])
    s1 = State(["0"])
    s2 = State(["1"])
    s3 = State(["3"])
    s4 = State(["4"])
    s5 = State(["5"])
    s6 = State(["10"])
    s8 = State(["9"])

    states: List[State] = [start_state, s0, s1, s2, s3, s4, s5, s6, s8, terminal_state]

    for i in range(1, len(states)):
        pTree.add_state(states[i])
        pTree.add_edge(states[i-1], states[i])

    assert graph_match_1.matches_unique_states(pTree)


def test_match_tree_2(graph_match_1: Graph):
    start_state = State([""])
    terminal_state = State([""], is_terminal=True)
    pTree = PrefixTree(start_state)
    s0 = State(["0"])
    s1 = State(["0"])
    s2 = State(["1"])
    s3 = State(["6"])
    s4 = State(["7"])
    s6 = State(["10"])

    states: List[State] = [start_state, s0, s1, s2, s3, s4, s6, terminal_state]

    for i in range(1, len(states)):
        pTree.add_state(states[i])
        pTree.add_edge(states[i-1], states[i])

    assert not graph_match_1.matches_unique_states(pTree)


def test_get_state_by_id(graph: Graph):
    state = graph.states[0]

    assert graph.get_state_by_id(state.state_id) == state


def test_get_state_by_hash_incorrect(graph: Graph):
    state = State(["other"])

    with pytest.raises(StateDoesNotExistException):
        graph.get_state_by_id(state.state_id)


def test_add_state(graph: Graph):
    new_state = State(["5"])

    assert len(graph.states) == 5
    graph.add_state(new_state)

    assert len(graph.states) == 6
    assert new_state in graph.states.values()
    assert graph.get_state_by_id(new_state.state_id) == new_state


def test_add_state_properties_pointers():
    graph = Graph()
    state1 = State(["prop"])
    state2 = State(["prop"])

    graph.add_state(state1)
    graph.add_state(state2)

    assert graph.size() == 2
    assert id(state1.properties) == id(state2.properties)
    assert state1.state_id != state2.state_id


def test_add_edge(graph: Graph):
    state1 = graph.states[1]
    state2 = graph.states[2]
    assert state2 not in graph.get_outgoing_states(state1)

    graph.add_edge(state1, state2, EdgeProperties())

    assert state2 in graph.get_outgoing_states(state1)


def test_add_edge_incorrect(graph: Graph):
    state1 = State(["other"])

    assert graph.add_edge(state1, graph.states[0], EdgeProperties()) is False
    assert graph.add_edge(graph.states[0], state1, EdgeProperties()) is False


def test_size(graph: Graph):
    assert graph.size() == len(graph.states)


def test_get_outgoing_states(graph: Graph):
    state0 = graph.states[0]

    assert len(graph.get_outgoing_states(state0)) == 3
    assert graph.states[1] in graph.get_outgoing_states(state0)


def test_merge_states(graph: Graph):
    state1 = graph.states[1]
    state3 = graph.states[3]

    graph.merge_states(state1, state3)

    assert len(graph.states) == 4
    assert state1.properties.log_templates == [["1"], ["3"]]
    assert state3.state_id not in graph.states.keys()
    assert graph.get_outgoing_states(state1)[0] == state1


def test_merge_states2(graph: Graph):
    state0 = graph.states[0]
    state3 = graph.states[3]

    assert graph.start_node == state0
    graph.merge_states(state3, state0)

    assert len(graph.states) == 4
    assert state3.properties.log_templates == [["3"], ["0"]]
    assert state0.state_id not in graph.states.keys()
    outgoing = graph.get_outgoing_states(state3)
    assert graph.states[1] in outgoing
    assert state3 not in graph.get_outgoing_states(state3)
    assert graph.get_outgoing_states(graph.states[1])[0] == state3
    assert graph.start_node == state3


def test_merge_states3(graph: Graph):
    state2 = graph.states[2]
    state4 = graph.states[4]

    graph.merge_states(state2, state4)

    assert len(graph.states) == 4
    assert state2.properties.log_templates == [["2"], ["4"]]
    assert state4.state_id not in graph.states.keys()
    assert state2 in graph.get_outgoing_states(graph.states[0])


def test_complex_merge_1(fully_connected_graph: Graph):
    fully_connected_graph.merge_states(fully_connected_graph.states[0],
                                       fully_connected_graph.states[1])

    new_node = fully_connected_graph.states[0]

    assert len(fully_connected_graph.states) == 3
    assert new_node.properties.log_templates == [["0"], ["1"]]

    for state in fully_connected_graph.states.values():
        for other_state in fully_connected_graph.states.values():
            assert state in fully_connected_graph.get_outgoing_states(other_state)


def test_complex_merge_2(graph_2: Graph):
    graph_2.full_merge_states(graph_2.states[0],
                              graph_2.states[1])

    new_node = graph_2.states[4]
    
    assert len(graph_2) == 3
    assert new_node.properties.log_templates == [["0"], ["1"]]


def test_complex_merge_3(graph_2: Graph):
    graph_2.full_merge_states(graph_2.states[0],
                              graph_2.states[3])

    assert len(graph_2) == 1
    new_node = list(graph_2.states.values())[0]
    assert sorted(new_node.properties.log_templates) == sorted([["0"], ["1"], ["2"]])


def test_complex_merge_4(graph_2: Graph):
    s5: State = State(["1"])
    graph_2.add_state(s5)
    graph_2.add_edge(s5, s5)
    graph_2.add_edge(s5, graph_2.states[0])

    graph_2.full_merge_states(graph_2.states[0],
                              graph_2.states[3])
    assert len(graph_2) == 1
    new_node = list(graph_2.states.values())[0]
    assert sorted(new_node.properties.log_templates) == sorted([["0"], ["1"], ["2"]])


def test_complex_merge_5(graph_2: Graph):
    s5: State = State(["1"])
    s6: State = State(["2"])
    s7: State = State(["0"])

    graph_2.add_state(s5)
    graph_2.add_state(s6)
    graph_2.add_state(s7)

    graph_2.add_edge(s5, s5)
    graph_2.add_edge(s5, graph_2.states[0])
    graph_2.add_edge(s5, s6)
    graph_2.add_edge(s7, s6)
    graph_2.add_edge(s7, s7)

    graph_2.full_merge_states(graph_2.states[0],
                              graph_2.states[3])

    new_node = list(graph_2.states.values())[0]
    assert len(graph_2) == 1
    assert sorted(new_node.properties.log_templates) == sorted([["0"], ["1"], ["2"]])


def test_merge_equivalent_children_self():
    state0 = State(["0"])
    state1 = State(log_templates=[["1"], ["0"]])

    graph: Graph = Graph()

    graph.add_state(state0)
    graph.add_state(state1)

    graph.add_edge(state0, state1)
    graph.add_edge(state0, state0)

    graph.merge_equivalent_children(state0)

    assert len(graph) == 1
