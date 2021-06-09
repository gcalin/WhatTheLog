import pytest

from whatthelog.exceptions import InvalidTreeException
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State

@pytest.fixture()
def tree():
    return PrefixTree(None, State(["root"]))


def test_add_child(tree: PrefixTree):

    root = tree.get_root()
    state = State(["state1"])
    tree.add_child(state, root)

    assert len(tree) == 2

    assert len(tree.get_children(root)) == 1
    assert tree.get_children(root)[0] == state


def test_add_child_incorrect_parent(tree: PrefixTree):
    with pytest.raises(AssertionError):
        fake_root = State(["fake"])
        child = State(["child"])

        tree.add_child(child, fake_root)


def test_get_root():
    root = State(["root"])
    tree = PrefixTree(None, root)

    assert tree.get_root() == root


def test_get_children(tree: PrefixTree):
    child1 = State(["child1"])
    child2 = State(["child2"])

    assert tree.get_children(tree.get_root()) == []

    tree.add_child(child1, tree.get_root())
    tree.add_child(child2, tree.get_root())

    assert len(tree.get_children(tree.get_root())) == 2
    assert tree.get_children(tree.get_root())[0] == child1
    assert tree.get_children(tree.get_root())[1] == child2


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


def test_add_branch(tree: PrefixTree):
    other = PrefixTree(None, State(["other"]))
    child1 = State(["child1"])
    other.add_child(child1, other.get_root())

    tree.add_branch(other.get_root(), other, tree.get_root())

    assert len(tree.get_children(tree.get_root())) == 1
    assert len(tree.get_children(tree.get_children(tree.get_root())[0])) == 1


def test_merge(tree: PrefixTree):
    tree.add_child(State(["child1"]), tree.get_root())
    other = PrefixTree(None, tree.get_root())
    other.add_child(State(["child2"]), other.get_root())

    tree.merge(other)

    assert len(tree.get_children(tree.get_root())) == 2


def test_merge_complex(tree: PrefixTree):
    child1 = State(["child1"])

    tree.add_child(child1, tree.get_root())
    other = PrefixTree(None, tree.get_root())
    other.add_child(child1, other.get_root())
    other.add_child(State(["child2"]), child1)
    other.add_child(State(["child3"]), child1)

    tree.merge(other)

    tree.merge(other)

    assert len(tree.get_children(tree.get_root())) == 1
    assert tree.get_children(tree.get_root())[0] == child1
    assert len(tree.get_children(child1)) == 2


def test_merge_different_roots(tree: PrefixTree):
    with pytest.raises(InvalidTreeException):
        other = PrefixTree(None, State(["other"]))
        tree.merge(other)
