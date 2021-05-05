import pytest

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
    assert len(tree.edges) == 1

    assert len(tree.get_children(root)) == 1
    assert tree.get_children(root)[0] == state
