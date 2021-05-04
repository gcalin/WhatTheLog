from typing import List, Tuple

from whatthelog.exceptions import StateDoesNotExistException
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.graph import Graph
import pytest

from whatthelog.prefixtree.state import State


@pytest.fixture
def graph():

    graph = Graph()
    state0 = State(["0"])
    state1 = State(["1"])
    state2 = State(["2"])
    state3 = State(["3"])
    state4 = State(["4"])

    states = [state0, state1, state2, state3, state4]

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
    assert new_state in graph.states
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
