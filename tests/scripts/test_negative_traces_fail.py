import os
from typing import Tuple, Dict, List

from scripts.log_scrambler import process_file
from scripts.match_trace import match_trace
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree

test_logs_dir = 'tests/resources/testlogs/'

test_logs = [
    'xx1',
    'xx2',
    'xx3',
    'xx4',
    'xx5',
    'xx6',
    'xx7',
    'xx8',
]


def prefix_tree() -> PrefixTree:
    """
    Generates a prefix tree based on real log data.
    :return: Tuple of a PrefixTree and a dictionary mapping log templates
     to unique ids.
    """
    return PrefixTreeFactory.get_prefix_tree(test_logs_dir, "resources/config.json")


def get_syntax_tree() -> SyntaxTree:
    """
    Gets the syntax tree with real configurations
    :return: SyntaxTree
    """
    return Parser().parse_file("resources/config.json")


def generate_negative_traces(filename: str) -> List[str]:
    """
    Generate negative trace
    :param filename: the filename which contains the log
    :return: a list containing all log statements of a negative log trace.
    """
    tree: SyntaxTree = get_syntax_tree()
    process_file(filename, filename + '_output', tree)
    with open(filename + '_output', 'r') as f:
        return f.read().splitlines()


def delete_negative_traces(filename) -> None:
    """
    Remove negative trace
    :param filename: the filename which contains the negative trace
    """
    os.remove(filename + '_output')


def execute_test_on_trace(filename) -> None:
    """
    This function does three things.
        1. generate negative traces
        2. validate whether these traces are indeed invalid.
        3. clean up the traces.

    :param filename: the filename which contains the log
    """
    # Generate wrong traces and retrieve them
    wrong_trace = generate_negative_traces(filename)

    # Check whether this trace is invalid
    result = match_trace(prefix_tree(), wrong_trace, get_syntax_tree())
    assert result is None, 'Negative trace not identified as invalid'

    # Clean up
    delete_negative_traces(filename)


def test_negative_traces() -> None:
    """
    Tests the functionality of negative traces on real data.
    """
    for t in test_logs:
        # We try to make negative traces of every log file.
        for i in range(10):
            # We make 10 negative traces for every log file, as they are created randomly
            execute_test_on_trace(test_logs_dir + t)
