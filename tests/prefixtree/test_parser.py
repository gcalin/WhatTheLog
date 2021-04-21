from whatthelog.prefixtree.parser import Parser
from whatthelog.prefixtree.prefix_tree import PrefixTree


def test_parse_file_simple():
    tree = Parser().parse_file("tests/resources/test_simple.json")
    expected = PrefixTree("node", "[node]", False)

    assert tree == expected


def test_parse_file_complex():
    tree = Parser().parse_file("tests/resources/test.json")

    expected = PrefixTree("root", "[root]", False)
    node1 = PrefixTree("node1", "[node1]", False)
    node2 = PrefixTree("node2", r"(\d{3}) (\d{2})", True)
    node3 = PrefixTree("node3", "[node3]", False)

    node2.insert(node3)

    expected.insert(node1)
    expected.insert(node2)

    assert tree == expected
