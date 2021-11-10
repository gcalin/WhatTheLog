import os
from pathlib import Path

import pytest

from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


@pytest.fixture
def abbadingo_tree() -> PrefixTree:
    traces_path = "tests/resources/abbadingo_input_1.a"
    tree = PrefixTreeFactory().get_prefix_tree(
        traces_path,
        "tests/resources/config.json",
        one_state_per_template=True,
        abbadingo_format=True)
    return tree


def test_multiple_traces(abbadingo_tree):
    traces_path = "tests/resources/abbadingo_input_1.a"
    with open(traces_path) as f:
        traces = f.readlines()[1:]
    for trace in traces:
        res: bool = abbadingo_tree.matches_abbadingo_format(trace.rstrip())
        assert res




