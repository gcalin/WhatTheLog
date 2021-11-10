import pytest

from whatthelog.syntaxtree.syntax_tree import SyntaxTree

@pytest.fixture()
def tree():
    return SyntaxTree("node", "[node]", False)


def test_constructor():
    tree = SyntaxTree("node", "[name]", False)

    assert tree.name == "node"
    assert tree.prefix == "[name]"
    assert tree.isRegex == False


def test_insert(tree):
    assert len(tree.get_children()) == 0
    tree.insert(SyntaxTree("childNode", "[childNode]", False))
    assert len(tree.get_children()) == 1


def test_search_root(tree):
    assert tree.search("[node]").name == tree.name


def test_invalid_search(tree):
    assert tree.search("[invalid]") is None


def test_search_child_simple(tree):
    nodeLeft = SyntaxTree("nodeLeft", "[nodeLeft]", False)
    nodeRight = SyntaxTree("nodeRight", "[nodeRight]", False)
    tree.insert(nodeLeft)
    tree.insert(nodeRight)

    assert tree.search("[node][nodeLeft]").name == nodeLeft.name


def test_search_regex(tree):
    nodeLeft = SyntaxTree("nodeLeft", r"\[regexNode1+\]", True)
    nodeRight = SyntaxTree("nodeRight", "[nodeRight]", False)
    tree.insert(nodeLeft)
    tree.insert(nodeRight)

    assert tree.search("[node][regexNode1]").name == nodeLeft.name


def test_search_regex_invalid(tree):
    nodeLeft = SyntaxTree("nodeLeft", r"\[regexNode1+\]", True)
    nodeRight = SyntaxTree("nodeRight", "[nodeRight]", False)
    tree.insert(nodeLeft)
    tree.insert(nodeRight)

    assert tree.search("[node][regexNode]") is None
