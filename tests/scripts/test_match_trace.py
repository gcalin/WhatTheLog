import random
import pytest

from scripts.match_trace import match_trace
from whatthelog.prefixtree.prefix_tree import PrefixTree, State
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from typing import List, Dict


@pytest.fixture()
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

    p_tree: PrefixTree = PrefixTree(t0)
    p_tree.add_child(t2, t0)
    p_tree.add_child(t1, t0)
    p_tree.add_child(t3, t2)

    # t0: PrefixTree = PrefixTree(State([0]), None)
    # t1: PrefixTree = PrefixTree(State([1, 2]), t0)
    # t0.add_child(t1)
    # t2: PrefixTree = PrefixTree(State([3]), t1)
    # t0.add_child(t2)
    # t3: PrefixTree = PrefixTree(State([4, 6, 5]), t2)
    # t2.add_child(t3)

    """
            Graph structure:
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


def test_match_trace_empty_trace(state_tree, traces_t0, syntax_tree):
    """
    Tests the match_trace function on an empty trace
    """
    trace = []
    expected_result = []

    res = match_trace(state_tree, trace, syntax_tree)
    assert res == expected_result, "Non-empty result " + res.__str__()


def test_match_trace_fail_root(state_tree, traces_t0, syntax_tree):
    """
    Tests the match_trace function on a trace that fails to match in the first line
    """
    expected_result = None

    for count, t in enumerate(traces_t0):
        t[0] = "fail" + t[0]
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_result == res, "Succeeded root matching on an illegal trace"


def test_match_trace_root(state_tree, traces_t0, syntax_tree):
    """
    Tests the match_trace function on a trace that succeeds with exactly 1 line
    """
    expected_path = [[state_tree.get_root()]]
    for count, t in enumerate(traces_t0):
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_path[count - 1] == res, "Failed to match the first line to the root"


def test_match_trace_traversal_1(state_tree, traces_t1, syntax_tree):
    """
    Tests the match_trace function on an accepted longer trace
    """
    expected_path = [[state_tree.get_root(),
                      state_tree.get_children(state_tree.get_root())[0]]] * 2
    for count, t in enumerate(traces_t1):
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_path[count - 1] == res, "Failed multi-state traversal"


def test_match_trace_fail_traversal_2(state_tree, traces_t2, syntax_tree):
    """
    Tests the match_trace function on an a trace that fails at its second line
    """
    expected_result = None

    for count, t in enumerate(traces_t2):
        t[1] = "fail" + t[1]
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_result == res, "Succeeded multi-state traversal on an illegal trace"


def test_match_trace_single_traversal_2(state_tree, traces_t2, syntax_tree):
    """
    Tests the match_trace function on an accepted longer trace
    """
    expected_path = [[state_tree.get_root(),
                      state_tree.get_children(state_tree.get_root())[1]]]
    for count, t in enumerate(traces_t2):
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_path[count - 1] == res, "Failed multi-state traversal"


def test_match_trace_fail_traversal_3(state_tree, traces_t3, syntax_tree):
    """
    Tests the match_trace function on a trace that fails to match at a later line
    """
    expected_result = None

    for count, t in enumerate(traces_t3):
        t[2] = "fail" + t[2]
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_result == res, "Succeeded multi-state traversal on an illegal trace"


def test_match_trace_traversal_3(state_tree, traces_t3, syntax_tree):
    """
    Tests the match_trace function on an accepted longer trace
    """
    root: State = state_tree.get_root()
    t2 = state_tree.get_children(root)[1]
    t3 = state_tree.get_children(t2)[0]

    expected_path = [[root, t2, t3]] * 3
    for count, t in enumerate(traces_t3):
        res = match_trace(state_tree, t, syntax_tree)
        assert expected_path[count - 1] == res, "Failed multi-state traversal"


def test_match_trace_no_successor_root(state_tree, syntax_tree):
    """
    Tests the match_trace function on a trace that has no successor state after the first line
    """
    trace = ["[p0][p1]", "[p0][p4][p0]"]    # 0 - 3: Invalid transition
    expected_result = None

    res = match_trace(state_tree, trace, syntax_tree)
    assert expected_result == res, "Non-empty result" + res.__str__()


def test_match_trace_no_successor_rec_1(state_tree, syntax_tree):
    """
    Tests the match_trace function on a trace that has no successor state after the second line
    """
    trace = ["[p0][p1]", "[p0][p4][p1]", "[p0][p1]"]   # 0 - 3: Invalid transition
    expected_result = None

    res = match_trace(state_tree, trace, syntax_tree)
    assert expected_result == res, "Non-empty result" + res.__str__()


def test_match_trace_no_successor_rec_2(state_tree, syntax_tree):
    """
    Tests the match_trace function on a trace that has no successor state after a later line
    """
    root: State = state_tree.get_root()
    t2: State = state_tree.get_children(root)[1]
    t3: State = state_tree.get_children(t2)[0]
    t4: State = State(["p0p1"])
    state_tree.add_child(t4, t3)
    trace = ["[p0][p1]", "[p0][p4][p1]", "[p0][p4][p0]", "[p0][p4][p1]"]
    expected_result = None

    res = match_trace(state_tree, trace, syntax_tree)
    assert expected_result == res, "Non-empty result" + res.__str__()
