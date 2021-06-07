import pytest

from whatthelog.exceptions import InvalidTreeException
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State

@pytest.fixture()
def tree():
    return PrefixTree(State(["root"]))


def test_add_child(tree: PrefixTree):

    root = tree.get_root()
    state = State(["state1"])
    tree.add_child(state, root)

    assert len(tree) == 2
    assert len(tree.outgoing_edges) == 1
    assert len(tree.incoming_edges) == 1

    assert len(tree.get_children(root)) == 1
    assert tree.get_children(root)[0] == state


def test_add_child_incorrect_parent(tree: PrefixTree):
    with pytest.raises(AssertionError):
        fake_root = State(["fake"])
        child = State(["child"])

        tree.add_child(child, fake_root)


def test_get_root():
    root = State(["root"])
    tree = PrefixTree(root)

    assert tree.get_root() == root


def test_get_children(tree: PrefixTree):
    child1 = State(["child1"])
    child2 = State(["child2"])

    assert tree.get_children(tree.get_root()) == []

    tree.add_child(child1, tree.get_root())
    tree.add_child(child2, tree.get_root())

    assert len(tree.get_children(tree.get_root())) == 2
    assert child1 in tree.get_children(tree.get_root())
    assert child2 in tree.get_children(tree.get_root())


def test_get_parent(tree: PrefixTree):
    root = tree.get_root()
    child1 = State(["child1"])

    tree.add_child(child1, root)

    assert tree.get_parent(child1) == root


def test_get_parent_not_in_tree(tree: PrefixTree):
    root = tree.get_root()
    child1 = State(["child1"])

    with pytest.raises(AssertionError):
        tree.get_parent(child1)


def test_get_parent_of_root(tree: PrefixTree):
    assert tree.get_parent(tree.get_root()) is None