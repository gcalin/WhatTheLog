import random

import pytest

from scripts import log_scrambler
from typing import List

from whatthelog.syntaxtree.syntax_tree import SyntaxTree


@pytest.fixture()
def syntax_tree() -> SyntaxTree:
    t: SyntaxTree = SyntaxTree("root", "[root]", False)
    t.insert(SyntaxTree("p1", "[p1]", False))
    t.insert(SyntaxTree("p2", "[p2]", False))
    t.insert(SyntaxTree("p3", "[p3]", False))

    assert t.search("[root][p1]") == SyntaxTree("p1", "[p1]", False), "Invalid Prefix Tree implementation"
    assert t.search("[root][p2]") == SyntaxTree("p2", "[p2]", False), "Invalid Prefix Tree implementation"
    assert t.search("[root][p3]") == SyntaxTree("p3", "[p3]", False), "Invalid Prefix Tree implementation"

    return t


@pytest.fixture()
def lines() -> List[str]:
    l: List[str] = ["[root][p3]"]
    l.extend(["[root][p1]"] * 3)
    l.extend(["[root][p2]"] * 5)
    l.append("[root][p1]")
    l.extend(["[root][p2]"] * 2)
    l.append("[root][p3]")
    return l


def test_get_section_invalid(syntax_tree, lines):
    """
    Tests that the delete function works on empty files
    """

    # Test negative index
    res1 = log_scrambler.get_section(lines, -1, syntax_tree)
    assert res1 == []

    # Test index too large
    lines = []
    res2 = log_scrambler.get_section(lines, 42, syntax_tree)

    assert res2 == []


def test_get_section_beginning(syntax_tree, lines):
    """
    Tests the get_section function by starting in the first line of a section
    """

    section1_expected = [1, 2, 3]
    section2_expected = [4, 5, 6, 7, 8]
    section3_expected = [10, 11]

    section1: List[int] = log_scrambler.get_section(lines, 1, syntax_tree)
    assert section1 == section1_expected, "Wrong section for " + lines[1]

    section2: List[int] = log_scrambler.get_section(lines, 4, syntax_tree)
    assert section2 == section2_expected, "Wrong section for " + lines[4]

    section3: List[int] = log_scrambler.get_section(lines, 10, syntax_tree)
    assert section3 == section3_expected, "Wrong section for " + lines[4]


def test_get_section_middle(syntax_tree, lines):
    """
    Tests the get_section function by starting in the middle of a section
    """

    section1_expected = [1, 2, 3]
    section2_expected = [4, 5, 6, 7, 8]

    section1: List[int] = log_scrambler.get_section(lines, 2, syntax_tree)
    assert section1 == section1_expected, "Wrong section for " + lines[2]

    section2: List[int] = log_scrambler.get_section(lines, 7, syntax_tree)
    assert section2 == section2_expected, "Wrong section for " + lines[7]


def test_get_section_end(syntax_tree, lines):
    """
    Tests the get_section function by starting in the last line of a section
    """

    section1_expected = [1, 2, 3]
    section2_expected = [4, 5, 6, 7, 8]
    section3_expected = [10, 11]

    section1: List[int] = log_scrambler.get_section(lines, 3, syntax_tree)
    assert section1 == section1_expected, "Wrong section for " + lines[1]

    section2: List[int] = log_scrambler.get_section(lines, 8, syntax_tree)
    assert section2 == section2_expected, "Wrong section for " + lines[4]

    section3: List[int] = log_scrambler.get_section(lines, 11, syntax_tree)
    assert section3 == section3_expected, "Wrong section for " + lines[4]


def test_get_section_only_line(syntax_tree, lines):
    """
    Tests the get_section function on sections made of exactly one line
    """

    section0_expected = [0]
    section1_expected = [9]
    section2_expected = [len(lines) - 1]

    section0: List[int] = log_scrambler.get_section(lines, 0, syntax_tree)
    assert section0 == section0_expected, "Wrong section for " + lines[0]

    section1: List[int] = log_scrambler.get_section(lines, 9, syntax_tree)
    assert section1 == section1_expected, "Wrong section for " + lines[9]

    section2: List[int] = log_scrambler.get_section(lines, len(lines) - 1, syntax_tree)
    assert section2 == section2_expected, "Wrong section for " + lines[-1]


def test_get_section_2_line_file(syntax_tree, lines):
    """
    Tests the get_section function on a very short file
    """

    lines = [lines[0]] * 2
    lines1 = lines.copy()

    section1_expected = [0, 1]

    section1: List[int] = log_scrambler.get_section(lines, 0, syntax_tree)
    assert section1 == section1_expected, "Wrong section for " + lines[0]

    section2: List[int] = log_scrambler.get_section(lines1, 1, syntax_tree)
    assert section2 == section1_expected, "Wrong section for " + lines1[1]


def test_delete_single_entry(syntax_tree, lines):
    """
    Tests the delete function on a single entry
    """
    line_number: int = 9

    # Set up mock
    choice = random.choice
    random.choice = lambda _: line_number

    # Get initial line and size
    line: str = lines[line_number]
    initial_size: int = len(lines)

    log_scrambler.delete_one(lines, syntax_tree)

    random.choice = choice

    # Assert that the line was deleted and the size changed
    assert initial_size - 1 == len(lines), "Delete function failed: same number of lines before and after"
    assert lines[line_number] != line, "Delete function failed: line unchanged"


def test_delete_single_entry_multiple_times(syntax_tree, lines):
    """
    Tests the delete function on a single entry multiple times
    """

    # Get initial line and size
    first_line = lines[0]
    initial_size = len(lines)

    choice = random.choice

    for count, line in enumerate([9, 0, len(lines) - 3]):
        # Set up mock
        random.choice = lambda _: line

        # Get the first line and the size
        deleted_line = lines[line]

        # Delete the line
        log_scrambler.delete_one(lines, syntax_tree)

        random.choice = choice

        # Assert that the line was deleted and the size changed
        assert initial_size - count - 1 == len(lines), "Delete function failed: same number of lines before and after"
        assert len(lines) == 0 or lines[
            min(line, len(lines) - 1)] != deleted_line, "Delete function failed: line unchanged"

    assert first_line not in lines, "Delete function failed: first and last lines were not deleted properly"


def test_delete_section_entry(syntax_tree, lines):
    """
    Tests the delete function on a single entry
    """
    line_number: int = 1

    # Set up mock
    choice = random.choice
    random.choice = lambda _: line_number

    # Get initial line and size
    line: str = lines[line_number]
    initial_size: int = len(lines)

    log_scrambler.delete_one(lines, syntax_tree)

    random.choice = choice

    # Assert that the line was deleted and the size changed
    assert initial_size - 3 == len(lines), "Delete function failed: same number of lines before and after"
    assert lines[line_number] != line, "Delete function failed: line in section unchanged"



def test_delete_section_multiple_times(syntax_tree, lines):
    """
    Tests the delete function on section entries multiple times
    """
    choice = random.choice

    for line, size in [(1, 3), (1, 5), (2, 2)]:
        # Get initial size
        initial_size = len(lines)

        # Set up mock
        random.choice = lambda _: line

        # Get the deleted line
        deleted_line = lines[line]

        # Delete the line
        log_scrambler.delete_one(lines, syntax_tree)

        random.choice = choice

        # Assert that the line was deleted and the size changed
        assert initial_size - size == len(lines), "Delete function failed: same number of lines before and after"
        assert len(lines) == 0 or lines[
            min(line, len(lines) - 1)] != deleted_line, "Delete function failed: line unchanged"


def test_delete_mixed_1(syntax_tree, lines):
    """
    Tests the delete function in a mixed scenario
    """
    choice = random.choice

    for line in [1, 6, 3, 0]:
        # Get initial size
        initial_size = len(lines)

        # Set up mock
        random.choice = lambda _: line

        # Get the line and the size
        deleted_line = lines[line]
        size = len(log_scrambler.get_section(lines, line, syntax_tree))

        # Delete the line
        log_scrambler.delete_one(lines, syntax_tree)

        random.choice = choice

        # Assert that the line was deleted and the size changed
        assert initial_size - size == len(lines), "Delete function failed: same number of lines before and after"
        assert len(lines) == 0 or lines[
            min(line, len(lines) - 1)] != deleted_line, "Delete function failed: line unchanged"

    assert len(lines) == 0, "Delete function failed: lines still remaining"


def test_swap_invalid(syntax_tree, lines):
    """
    Tests the swap function on a one-line file
    """
    lines = [lines[0]]
    initial = lines.copy()

    log_scrambler.swap(lines, syntax_tree)

    assert initial == lines


def test_swap_single_line(syntax_tree, lines):
    """
    Tests that swapping a single element does nothing
    """

    lines = [lines[0]]
    expected = lines.copy()

    log_scrambler.swap(lines, syntax_tree)

    assert expected == lines


def test_swap_simple_section(syntax_tree, lines):
    """
    Tests that swapping two adjacent sections is guaranteed when there are only two sections in the file
    """

    """
    A file of the form
    [AAA, AAA, AAA, BBB, BBB, BBB, BBB, BBB] should always result in [BBB, BBB, BBB, BBB, BBB, AAA, AAA, AAA]
    """
    lines = ([lines[0]] * 3 + [lines[1]] * 5)

    # Make a copy of the initial value of lines
    initial = lines.copy()

    # Create swapped version
    swapped = lines[3:] + lines[:3]

    log_scrambler.swap(lines, syntax_tree)

    # Assert correct swap
    assert swapped == lines

    # Swap for a second time
    log_scrambler.swap(lines, syntax_tree)

    # Check that the file is back to its initial form
    assert lines == initial


def test_swap_multiple_times(syntax_tree, lines):
    """
    Tests the swap function a series of times in a mixed scenario
    """
    initial = lines.copy()

    choice = random.choice

    # First swap sections 2 and 3
    random.choice = lambda _: 3
    expected: List[str] = ["[root][p3]"] \
                          + ["[root][p2]"] * 5 \
                          + ["[root][p1]"] * 4 \
                          + ["[root][p2]"] * 2 \
                          + ["[root][p3]"]
    log_scrambler.swap(lines, syntax_tree)
    random.choice = choice
    assert expected == lines

    # Swap sections 4 and 5
    random.choice = lambda _: 12
    expected: List[str] = ["[root][p3]"] \
                          + ["[root][p2]"] * 5 \
                          + ["[root][p1]"] * 4 \
                          + ["[root][p3]"] \
                          + ["[root][p2]"] * 2
    log_scrambler.swap(lines, syntax_tree)
    random.choice = choice
    assert expected == lines

    # Swap sections 3 and 4
    random.choice = lambda _: 9
    expected: List[str] = ["[root][p3]"] \
                          + ["[root][p2]"] * 5 \
                          + ["[root][p3]"] \
                          + ["[root][p1]"] * 4 \
                          + ["[root][p2]"] * 2
    log_scrambler.swap(lines, syntax_tree)
    random.choice = choice
    assert expected == lines

    # Swap sections 1 and 2
    random.choice = lambda _: 0
    expected: List[str] = ["[root][p2]"] * 5 \
                          + ["[root][p3]"] * 2 \
                          + ["[root][p1]"] * 4 \
                          + ["[root][p2]"] * 2
    log_scrambler.swap(lines, syntax_tree)
    random.choice = choice
    assert expected == lines
