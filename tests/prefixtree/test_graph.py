from copy import deepcopy
from typing import List

from whatthelog.exceptions import StateDoesNotExistException
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.adjacency_graph import AdjacencyGraph
import pytest

from whatthelog.prefixtree.state import State


@pytest.fixture
def graph():
    state0 = State(["0"])
    state1 = State(["1"])
    state2 = State(["2"])
    state3 = State(["3"])
    state4 = State(["4"])

    graph = AdjacencyGraph(None, state0)

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

    graph: AdjacencyGraph = AdjacencyGraph(None, states[0])

    for state in states[1:]:
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
    s4 = State(["0", "1"])

    states: List[State] = [s0, s1, s2, s3, s4]

    graph: AdjacencyGraph = AdjacencyGraph(None, states[0])

    for state in states[1:]:
        graph.add_state(state)

    graph.add_edge(s0, s0)
    graph.add_edge(s0, s1)
    graph.add_edge(s1, s2)
    graph.add_edge(s0, s3)
    graph.add_edge(s3, s4)

    return graph


def test_get_state_by_id(graph: AdjacencyGraph):
    state = graph.states[0]

    assert graph.get_state_by_id(id(state)) == state


def test_get_state_by_hash_incorrect(graph: AdjacencyGraph):
    state = State(["other"])

    with pytest.raises(StateDoesNotExistException):
        graph.get_state_by_id(id(state))


def test_add_state(graph: AdjacencyGraph):
    new_state = State(["5"])

    assert len(graph.states) == 5
    graph.add_state(new_state)

    assert len(graph.states) == 6
    assert new_state in graph.states.values()
    assert graph.get_state_by_id(id(new_state)) == new_state


def test_add_state_properties_pointers():
    state1 = State(["prop"])
    state2 = State(["prop"])

    graph = AdjacencyGraph(None, state1)
    graph.add_state(state2)

    assert len(graph) == 2
    assert id(state1.properties) == id(state2.properties)
    assert id(state1) != id(state2)


def test_add_edge(graph: AdjacencyGraph):
    state1 = graph.states[1]
    state2 = graph.states[2]
    assert state2 not in graph.get_outgoing_states(state1)

    graph.add_edge(state1, state2, EdgeProperties())

    assert state2 in graph.get_outgoing_states(state1)


def test_add_edge_incorrect(graph: AdjacencyGraph):
    state1 = State(["other"])

    assert graph.add_edge(state1, graph.states[0], EdgeProperties()) is False
    assert graph.add_edge(graph.states[0], state1, EdgeProperties()) is False


def test_size(graph: AdjacencyGraph):
    assert len(graph) == len(graph.states)


def test_get_outgoing_states(graph: AdjacencyGraph):
    state0 = graph.states[0]

    assert len(graph.get_outgoing_states(state0)) == 3
    assert graph.states[1] in graph.get_outgoing_states(state0)


def test_merge_states(graph: AdjacencyGraph):
    state1 = graph.states[1]
    state3 = graph.states[3]

    graph.merge_states(state1, state3)

    assert len(graph.states) == 4
    assert state1.properties.log_templates == ["1", "3"]
    assert graph.state_indices_by_id[id(state1)] == 1
    assert id(state3) not in graph.state_indices_by_id.keys()
    assert graph.get_outgoing_states(state1)[0] == state1


def test_merge_states2(graph: AdjacencyGraph):
    state0 = graph.states[0]
    state3 = graph.states[3]

    assert graph.start_node == state0
    graph.merge_states(state3, state0)

    assert len(graph.states) == 4
    assert state3.properties.log_templates == ["3", "0"]
    assert graph.state_indices_by_id[id(state3)] == 3
    assert id(state0) not in graph.state_indices_by_id.keys()
    assert graph.states[1] in graph.get_outgoing_states(state3)
    assert state3 not in graph.get_outgoing_states(state3)
    assert graph.get_outgoing_states(graph.states[1])[0] == state3
    assert graph.start_node == state3


def test_merge_states3(graph: AdjacencyGraph):
    state2 = graph.states[2]
    state4 = graph.states[4]

    graph.merge_states(state2, state4)

    assert len(graph.states) == 4
    assert state2.properties.log_templates == ["2", "4"]
    assert graph.state_indices_by_id[id(state2)] == 2
    assert id(state4) not in graph.state_indices_by_id.keys()
    assert state2 in graph.get_outgoing_states(graph.states[0])
