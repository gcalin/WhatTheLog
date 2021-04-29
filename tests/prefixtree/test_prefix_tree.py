import pytest

from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.exceptions import InvalidTreeException

@pytest.fixture()
def tree():
    return PrefixTree(State([""]))

@pytest.fixture()
def another_tree():
    return PrefixTree(State([""]))

def test_merge_same_tree(tree: PrefixTree, another_tree: PrefixTree):

    root1 = tree.get_root()
    child1 = State(["A"])
    child2 = State(["B"])
    child3 = State(["C"])
    tree.add_child(child1, root1)
    tree.add_child(child2, child1)
    tree.add_child(child3, child2)

    root2 = another_tree.get_root()
    child4 = State(["A"])
    child5 = State(["B"])
    child6 = State(["C"])
    another_tree.add_child(child4, root2)
    another_tree.add_child(child5, child4)
    another_tree.add_child(child6, child5)

    tree.merge(another_tree)

    assert tree.size() == 4

def test_merge_separate(tree: PrefixTree, another_tree: PrefixTree):

    root1 = tree.get_root()
    child1 = State(["A"])
    child2 = State(["B"])
    child3 = State(["C"])
    tree.add_child(child1, root1)
    tree.add_child(child2, child1)
    tree.add_child(child3, child2)

    root2 = another_tree.get_root()
    child4 = State(["D"])
    child5 = State(["E"])
    child6 = State(["F"])
    another_tree.add_child(child4, root2)
    another_tree.add_child(child5, child4)
    another_tree.add_child(child6, child5)

    tree.merge(another_tree)

    assert tree.size() == 7
    assert len(tree.get_children(root1)) == 2
    for child in [child1, child2, child4, child5]:
        assert len(tree.get_children(child)) == 1

def test_merge_split(tree: PrefixTree, another_tree: PrefixTree):

    root1 = tree.get_root()
    child1 = State(["A"])
    child2 = State(["B"])
    child3 = State(["C"])
    tree.add_child(child1, root1)
    tree.add_child(child2, child1)
    tree.add_child(child3, child2)

    root2 = another_tree.get_root()
    child4 = State(["A"])
    child5 = State(["B"])
    child6 = State(["D"])
    another_tree.add_child(child4, root2)
    another_tree.add_child(child5, child4)
    another_tree.add_child(child6, child5)

    tree.merge(another_tree)

    assert tree.size() == 5
    assert len(tree.get_children(child2)) == 2
    assert tree.get_parent(child6) == child2

def test_merge_different_roots(tree: PrefixTree):

    a_different_tree = PrefixTree(State(["not an empty string"]))

    with pytest.raises(InvalidTreeException):
        tree.merge(a_different_tree)
