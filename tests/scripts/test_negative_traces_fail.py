from pathlib import Path
import os
from typing import List

from scripts.log_scrambler import process_file
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory

project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
test_logs_dir = project_root.joinpath('tests/resources/testlogs/')

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
iterations = 10


def prefix_tree() -> PrefixTree:
    """
    Generates a prefix tree based on real log data.
    :return: Tuple of a PrefixTree and a dictionary mapping log templates
     to unique ids.
    """
    return PrefixTreeFactory.get_prefix_tree(test_logs_dir, project_root.joinpath("resources/config.json"))


def get_syntax_tree() -> SyntaxTree:
    """
    Gets the syntax tree with real configurations
    :return: SyntaxTree
    """
    return SyntaxTreeFactory().parse_file(project_root.joinpath("resources/config.json"))


def generate_negative_traces(filename: str) -> List[str]:
    """
    Generate negative trace
    :param filename: the filename which contains the log
    :return: a list containing all log statements of a negative log trace.
    """

    tree: SyntaxTree = get_syntax_tree()
    process_file(filename, str(filename) + '_output', tree)
    with open(filename + '_output', 'r') as f:
        return f.read().splitlines()


def delete_negative_traces(filename) -> None:
    """
    Remove negative trace
    :param filename: the filename which contains the negative trace
    """
    os.remove(filename + '_output')


def execute_test_on_trace(filename: Path, state_tree: PrefixTree) -> int:
    """
    This function does three things.
        1. generate negative traces
        2. validate whether these traces are indeed invalid.
        3. clean up the traces.

    :param filename: the filename which contains the log
    """

    # Generate wrong traces and retrieve them
    wrong_trace = generate_negative_traces(str(filename))

    # Check whether this trace is invalid
    result = state_tree.match_trace(wrong_trace, debug=True)

    # Clean up
    delete_negative_traces(str(filename))

    return int(result == False)


def test_negative_traces() -> None:
    """
    Tests the functionality of negative traces on real data.
    """

    total = 0
    failed = 0

    state_tree: PrefixTree = prefix_tree()
    for t in test_logs:
        # We try to make negative traces of every log file.
        for i in range(iterations):
            total += 1
            # We make 10 negative traces for every log file, as they are created randomly
            failed += execute_test_on_trace(test_logs_dir.joinpath(t), state_tree)

    assert failed / total >= 0.9, "Less than 90% of the traces failed"
