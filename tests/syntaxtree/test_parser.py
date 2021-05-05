from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


def test_parse_file_simple():
    tree = SyntaxTreeFactory().parse_file("tests/resources/test_simple.json")
    expected = SyntaxTree("node", "[node]", False)

    assert tree == expected


def test_parse_file_complex():
    tree = SyntaxTreeFactory().parse_file("tests/resources/test.json")

    expected = SyntaxTree("root", "[root]", False)
    node1 = SyntaxTree("node1", "[node1]", False)
    node2 = SyntaxTree("node2", r"(\d{3}) (\d{2})", True)
    node3 = SyntaxTree("node3", "[node3]", False)

    node2.insert(node3)

    expected.insert(node1)
    expected.insert(node2)

    assert tree == expected
