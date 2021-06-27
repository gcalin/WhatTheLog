import pytest
from typing import List

from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.prefix_tree import PrefixTree, State
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.exceptions import UnidentifiedLogException
from whatthelog.auto_printer import AutoPrinter

def print(msg): AutoPrinter.static_print(msg)


def syntax_tree() -> SyntaxTree:
    t: SyntaxTree = SyntaxTree("p0", "[p0]", False)
    t.insert(SyntaxTree("p0p1", "[p1]", False))
    t.insert(SyntaxTree("p0p2", "[p2]", False))
    t.insert(SyntaxTree("p0p3", "[p3]", False))
    t.insert(SyntaxTree("p0p4", "[p4]", False))
    tp4 = t.search("[p0][p4]")
    tp4.insert(SyntaxTree("p0p4p0", "[p0]", False))
    tp4.insert(SyntaxTree("p0p4p1", "[p1]", False))
    tp4.insert(SyntaxTree("p0p4p2", "[p2]", False))
    tp4.insert(SyntaxTree("p0p4p3", "[p3]", False))
    tp4.insert(SyntaxTree("p0p4p5", "[p5]", False))

    assert t.search("[p0][p1]") == SyntaxTree("p0p1", "[p1]", False), "Invalid Prefix Tree implementation"
    assert t.search("[p0][p2]") == SyntaxTree("p0p2", "[p2]", False), "Invalid Prefix Tree implementation"
    assert t.search("[p0][p3]") == SyntaxTree("p0p3", "[p3]", False), "Invalid Prefix Tree implementation"

    return t


@pytest.fixture()
def state_tree() -> PrefixTree:
    t0: State = State(["p0p1"])
    t1: State = State(["p0p2", "p0p3"])
    t2: State = State(["p0p4p1"])
    t3: State = State(["p0p4p2", "p0p4p3", "p0p4p0"])

    p_tree: PrefixTree = PrefixTree(syntax_tree(), State([]))
    p_tree.add_child(t0, p_tree.get_root())
    p_tree.add_child(t2, t0)
    p_tree.add_child(t1, t0)
    p_tree.add_child(t3, t2)

    """
            Graph structure:
          root (empty)
              |
              |
            t0 (0)
            /    \
           /      \
          /        \
       t1 (1,2)  t2 (3)
                    |
                    |
                 t3 (4,5,6)
    """

    return p_tree


@pytest.fixture()
def traces_t0() -> List[List[str]]:
    # The root of the tree
    return [["[p0][p1]"]]  # 0


@pytest.fixture()
def traces_t1() -> List[List[str]]:
    # All possible ways to reach t1 in the state tree
    return [
        ["[p0][p1]", "[p0][p3]"],  # 0 - 1
        ["[p0][p1]", "[p0][p2]"]  # 0 - 2
    ]


@pytest.fixture()
def traces_t2() -> List[List[str]]:
    # All possible ways to reach t2 in the state tree
    return [
        ["[p0][p1]", "[p0][p4][p1]"]  # 0 - 3
    ]


@pytest.fixture()
def traces_t3() -> List[List[str]]:
    # All possible ways to reach t2 in the state tree
    return [
        ["[p0][p1]", "[p0][p4][p1]", "[p0][p4][p2]"],  # 0 - 3 - 4
        ["[p0][p1]", "[p0][p4][p1]", "[p0][p4][p3]"],  # 0 - 3 - 5
        ["[p0][p1]", "[p0][p4][p1]", "[p0][p4][p0]"]   # 0 - 3 - 6
    ]


def test_match_trace_empty_trace(state_tree):
    """
    Tests the match_trace function on an empty trace
    """

    assert not state_tree.match_trace([]), "Non-empty result!"


def test_match_trace_fail_root(state_tree, traces_t0):
    """
    Tests the match_trace function on a trace that fails to match in the first line
    """

    for count, t in enumerate(traces_t0):
        t[0] = "fail" + t[0]
        with pytest.raises(UnidentifiedLogException):
            state_tree.match_trace(t)


def test_match_trace_root(state_tree, traces_t0):
    """
    Tests the match_trace function on a trace that succeeds with exactly 1 line
    """

    t0 = state_tree.get_children(state_tree.get_root())[0]

    for count, t in enumerate(traces_t0):

        # As long as the root is not terminal, no path should exist
        assert not state_tree.match_trace(t), "Incorrectly matched a non-terminal path"

        # Make t0 terminal
        t0.is_terminal = True

        # There should now be a matching path that ends in a terminal state
        assert state_tree.match_trace(t, debug=True), "Failed to match the first line to the root"


def test_match_trace_traversal_1(state_tree, traces_t1):
    """
    Tests the match_trace function on an accepted longer trace
    """

    for count, t in enumerate(traces_t1):
        assert not state_tree.match_trace(t), "Failed multi-state traversal"


def test_match_trace_traversal_no_terminal_1(state_tree, traces_t1):
    """
    Tests the match_trace function on a real trace that does not end in a terminal state
    """

    t0 = state_tree.get_children(state_tree.get_root())[0]
    t1 = state_tree.get_children(t0)[1]

    t1.is_terminal = True

    for count, t in enumerate(traces_t1):
        assert state_tree.match_trace(t, True), "Failed multi-state traversal"


def test_match_trace_fail_traversal_2(state_tree, traces_t2):
    """
    Tests the match_trace function on an a trace that fails at its second line
    """

    for count, t in enumerate(traces_t2):
        t[1] = "fail" + t[1]
        with pytest.raises(UnidentifiedLogException):
            state_tree.match_trace(t)


def test_match_trace_single_traversal_no_terminal_2(state_tree, traces_t2):
    """
    Tests the match_trace function on an accepted longer trace that does not end in a terminal state
    """

    for count, t in enumerate(traces_t2):
        assert not state_tree.match_trace(t), "Trace matched without terminal state closure"


def test_match_trace_single_traversal_2(state_tree, traces_t2):
    """
    Tests the match_trace function on an accepted longer trace
    """

    t0 = state_tree.get_children(state_tree.get_root())[0]
    t2 = state_tree.get_children(t0)[0]
    t2.is_terminal = True

    for count, t in enumerate(traces_t2):
        assert state_tree.match_trace(t), "Failed multi-state traversal"


def test_match_trace_fail_traversal_3(state_tree, traces_t3):
    """
    Tests the match_trace function on a trace that fails to match at a later line
    """

    for count, t in enumerate(traces_t3):
        t[2] = "fail" + t[2]
        with pytest.raises(UnidentifiedLogException):
            state_tree.match_trace(t)


def test_match_trace_traversal_no_terminal_3(state_tree, traces_t3):
    """
    Tests the match_trace function on an accepted longer trace that does not end in a terminal state
    """

    for count, t in enumerate(traces_t3):
        assert not state_tree.match_trace(t), "Failed multi-state traversal"


def test_match_trace_traversal_3(state_tree, traces_t3):
    """
    Tests the match_trace function on an accepted longer trace
    """
    root: State = state_tree.get_root()
    t0 = state_tree.get_children(root)[0]
    t2 = state_tree.get_children(t0)[0]
    t3 = state_tree.get_children(t2)[0]
    t3.is_terminal = True

    for count, t in enumerate(traces_t3):
        assert state_tree.match_trace(t), "Failed multi-state traversal"


def test_match_trace_no_successor_root(state_tree):
    """
    Tests the match_trace function on a trace that has no successor state after the first line
    """

    trace = ["[p0][p1]", "[p0][p4][p0]"]    # 0 - 3: Invalid transition

    assert not state_tree.match_trace(trace), "Non-empty result!"


def test_match_trace_no_successor_rec_1(state_tree):
    """
    Tests the match_trace function on a trace that has no successor state after the second line
    """

    trace = ["[p0][p1]", "[p0][p4][p1]", "[p0][p1]"]   # 0 - 3: Invalid transition

    assert not state_tree.match_trace(trace), "Non-empty result!"


def test_match_trace_no_successor_rec_2(state_tree):
    """
    Tests the match_trace function on a trace that has no successor state after a later line
    """
    root: State = state_tree.get_root()
    t0 = state_tree.get_children(root)[0]
    t2: State = state_tree.get_children(t0)[0]
    t3: State = state_tree.get_children(t2)[0]
    t4: State = State(["p0p1"])
    state_tree.add_child(t4, t3)
    trace = ["[p0][p1]", "[p0][p4][p1]", "[p0][p4][p0]", "[p0][p4][p1]"]

    assert not state_tree.match_trace(trace), "Non-empty result!"
