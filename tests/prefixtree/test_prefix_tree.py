import pytest

from whatthelog.prefixtree.prefix_tree import PrefixTree

@pytest.fixture()
def tree():
    return PrefixTree("node", "[node]", False)


def test_constructor():
    tree = PrefixTree("node", "[name]", False)

    assert tree.name == "node"
    assert tree.prefix == "[name]"
    assert tree.isRegex == False

def test_insert(tree):
    assert len(tree.get_children()) == 0
    tree.insert(PrefixTree("childNode", "[childNode]", False))
    assert len(tree.get_children()) == 1


def test_search_root(tree):
    assert tree.search("[node]").name == tree.name


def test_invalid_search(tree):
    assert tree.search("[invalid]") is None


def test_search_child_simple(tree):
    nodeLeft = PrefixTree("nodeLeft", "[nodeLeft]", False)
    nodeRight = PrefixTree("nodeRight", "[nodeRight]", False)
    tree.insert(nodeLeft)
    tree.insert(nodeRight)

    assert tree.search("[node][nodeLeft]").name == nodeLeft.name


def test_search_regex(tree):
    nodeLeft = PrefixTree("nodeLeft", r"\[regexNode1+\]", True)
    nodeRight = PrefixTree("nodeRight", "[nodeRight]", False)
    tree.insert(nodeLeft)
    tree.insert(nodeRight)

    assert tree.search("[node][regexNode1]").name == nodeLeft.name


def test_search_regex_invalid(tree):
    nodeLeft = PrefixTree("nodeLeft", r"\[regexNode1+\]", True)
    nodeRight = PrefixTree("nodeRight", "[nodeRight]", False)
    tree.insert(nodeLeft)
    tree.insert(nodeRight)

    assert tree.search("[node][regexNode]") is None
