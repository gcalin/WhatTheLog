from typing import List, Tuple

from whatthelog.exceptions import StateDoesNotExistException
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.graph import Graph
import pytest

from whatthelog.prefixtree.state import State
from whatthelog.prefixtree.visualizer import Visualizer


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


def test_get_state_by_hash(graph: Graph):
    state = graph.states[0]

    assert graph.get_state_by_hash(hash(state)) == state


def test_get_state_by_hash_incorrect(graph: Graph):
    state = State(["other"])

    with pytest.raises(StateDoesNotExistException):
        graph.get_state_by_hash(hash(state))


def test_add_state(graph: Graph):
    new_state = State(["5"])

    assert len(graph.states) == 5
    graph.add_state(new_state)

    assert len(graph.states) == 6
    assert new_state in graph.states.values()
    assert graph.get_state_by_hash(hash(new_state)) == new_state


def test_add_state_properties_pointers():
    graph = Graph()
    state1 = State(["prop"])
    state2 = State(["prop"])

    graph.add_state(state1)
    graph.add_state(state2)

    assert graph.size() == 2
    assert id(state1.properties) == id(state2.properties)
    assert hash(state1) != hash(state2)


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


def test_get_outgoing_props():
    graph = Graph()
    state1 = State(["1"])
    state2 = State(["2"])
    graph.add_state(state1)
    graph.add_state(state2)

    props = EdgeProperties(["50"])
    graph.add_edge(state1, state2, props)

    assert graph.get_outgoing_props(state1) == [props]


def test_get_outgoing_states(graph: Graph):
    state0 = graph.states[0]

    assert len(graph.get_outgoing_states(state0)) == 3
    assert graph.states[1] in graph.get_outgoing_states(state0)


def test_merge_states(graph: Graph):
    state1 = graph.states[1]
    state3 = graph.states[3]

    graph.merge_states(state1, state3)

    assert len(graph.states) == 4
    assert state1.properties.log_templates == ["1", "3"]
    assert graph.state_indices_by_hash[hash(state1)] == 1
    assert hash(state3) not in graph.state_indices_by_hash.keys()
    assert graph.get_outgoing_states(state1)[0] == state1


def test_merge_states2(graph: Graph):
    state0 = graph.states[0]
    state3 = graph.states[3]

    assert graph.start_node == state0
    graph.merge_states(state3, state0)

    assert len(graph.states) == 4
    assert state3.properties.log_templates == ["3", "0"]
    assert graph.state_indices_by_hash[hash(state3)] == 3
    assert hash(state0) not in graph.state_indices_by_hash.keys()
    assert graph.states[1] in graph.get_outgoing_states(state3)
    assert state3 not in graph.get_outgoing_states(state3)
    assert graph.get_outgoing_states(graph.states[1])[0] == state3
    assert graph.start_node == state3


def test_merge_states3(graph: Graph):
    state2 = graph.states[2]
    state4 = graph.states[4]

    graph.merge_states(state2, state4)
    print(graph.edges.list)

    assert len(graph.states) == 4
    assert state2.properties.log_templates == ["2", "4"]
    assert graph.state_indices_by_hash[hash(state2)] == 2
    assert hash(state4) not in graph.state_indices_by_hash.keys()
    assert state2 in graph.get_outgoing_states(graph.states[0])


def get_graph_to_find_loops() -> Graph:
    state0 = State(["0"])
    state1 = State(["1"])
    state2 = State(["1"])
    state3 = State(["1"])
    state4 = State(["2"])
    state5 = State(["1"])
    state6 = State(["3"])
    state7 = State(["3"])
    state8 = State(["3"])
    state9 = State(["1"])

    graph = Graph(state0)

    graph.add_state(state0)
    graph.add_state(state1)
    graph.add_state(state2)
    graph.add_state(state3)
    graph.add_state(state4)
    graph.add_state(state5)
    graph.add_state(state6)
    graph.add_state(state7)
    graph.add_state(state8)
    graph.add_state(state9)

    graph.add_edge(state0, state1, EdgeProperties())
    graph.add_edge(state1, state2, EdgeProperties())
    graph.add_edge(state2, state3, EdgeProperties())
    graph.add_edge(state3, state4, EdgeProperties())
    graph.add_edge(state4, state5, EdgeProperties())
    graph.add_edge(state2, state6, EdgeProperties())
    graph.add_edge(state6, state7, EdgeProperties())
    graph.add_edge(state7, state8, EdgeProperties())
    graph.add_edge(state8, state9, EdgeProperties())

    return graph


def test_merge_loops_recurring():
    graph = get_graph_to_find_loops()

    assert len(graph.states) == 10
    graph.remove_loops(True)
    assert len(graph.states) == 4
    assert graph.get_outgoing_states(graph.states[1]) == [graph.states[6], graph.states[4], graph.states[1]]
    assert graph.get_outgoing_states(graph.states[4]) == [graph.states[1]]
    assert graph.get_outgoing_states(graph.states[6]) == [graph.states[1], graph.states[6]]


def test_merge_loops_singular():
    graph = get_graph_to_find_loops()

    assert len(graph.states) == 10
    graph.remove_loops()
    assert len(graph.states) == 7
    assert graph.get_outgoing_states(graph.states[1]) == [graph.states[6], graph.states[3], graph.states[1]]
    assert graph.get_outgoing_states(graph.states[4]) == [graph.states[5]]
    assert graph.get_outgoing_states(graph.states[6]) == [graph.states[6], graph.states[9]]
    assert graph.get_outgoing_states(graph.states[9]) == []
