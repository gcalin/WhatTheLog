import random
from typing import List, Union, Dict

from whatthelog.prefixtree.prefix_tree import PrefixTree, State
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.exceptions.unidentified_log_exception import \
    UnidentifiedLogException


def match_trace(
        prefix_tree: PrefixTree,
        log_templates: Dict[str, int],
        trace: List[str],
        syntax_tree: SyntaxTree) -> Union[List[State], None]:
    """
    Checks if a given trace matches a path in the prefix tree.
    First, the first line is checked against the node of the prefix tree.
    If it fits, a successor that matches the next line is picked and the list
    is traversed recursively.

    :param prefix_tree: The prefix tree in which states need to be matched.
    :param log_templates: A dictionary in which each template name is mapped to the id of
                          its corresponding prefix tree node.
    :param trace: The lines to be matched against the tree.
    :param syntax_tree: The syntax tree used to validate the lines.
    :return: If the trace corresponds to a sequence of states in the prefix tree,
             those states are returned in order. If no match is found, None is returned.
    """

    if len(trace) == 0:
        # If the trace is empty, then it has been fully parsed
        return []

    # Check which state the current line belongs to
    template = syntax_tree.search(trace[0])

    # If no state is found, raise an exception
    if template is None:
        raise UnidentifiedLogException(trace[0] + " was not identified as a valid log.")

    # Check if the template matches the root of the state tree
    if log_templates[template.name] in prefix_tree.state.log_ids:

        # If the trace is exactly one line long, return the root state
        if len(trace) == 1:
            return [prefix_tree.state]

        # Remove the checked line from trace
        trace[:] = trace[1:]

        # Find the state for the second line
        template = syntax_tree.search(trace[0])

        # If no state is found, raise an exception
        if template is None:
            raise UnidentifiedLogException(trace[0] + " was not identified as a valid log.")

        # Check if any of the children of current node contain the template in their state
        children: List[PrefixTree] = prefix_tree.get_children()
        successor = list(filter(lambda next_tree: log_templates[template.name] in next_tree.state.log_ids,
                                children))

        if len(successor) == 0:
            # If none found, the trace cannot be matched
            return None
        else:
            # Pick a random suitable next node
            # TODO: Should this be removed through an invariant in the prefix tree?
            next_node = random.choice(successor)

            # Continue the search recursively
            tail = match_trace_rec(next_node, log_templates, trace[1:], syntax_tree)

            if tail is None:
                # If the search failed, return none
                return None
            else:
                # If it was successful, prepend the current state
                tail.insert(0, prefix_tree.state)
                return tail


def match_trace_rec(
        prefix_tree: PrefixTree,
        log_templates: Dict[str, int],
        trace: List[str],
        syntax_tree: SyntaxTree) -> Union[List[State], None]:
    """
    Recursive helper for the match_trace function. This function recursively
    finds the next state in the prefix tree for a the first line in the trace
    if any exists.

    :param prefix_tree: The prefix tree in which states need to be matched.
    :param log_templates: A dictionary in which each template name is mapped to the id of
                          its corresponding prefix tree node.
    :param trace: The lines to be matched against the tree.
    :param syntax_tree: The syntax tree used to validate the lines.
    :return: If the trace corresponds to a sequence of states in the prefix tree,
             those states are returned in order. If no match is found, None is returned.
    """

    if len(trace) == 0:
        # If the trace is empty, then it has been fully parsed
        return [prefix_tree.state]

    # Find the template of the first line in the syntax tree
    template = syntax_tree.search(trace[0])

    # If no state is found, raise an exception
    if template is None:
        raise UnidentifiedLogException(trace[0] + " was not identified as a valid log.")

    # Check if any of the children of current node contain the template in their state
    children: List[PrefixTree] = prefix_tree.get_children()
    successor = list(filter(lambda next_tree: log_templates[template.name] in next_tree.state.log_ids,
                            children))

    if len(successor) == 0:
        # If none found, the trace cannot be matched
        return None
    else:
        # Pick a random suitable next node
        next_node = random.choice(successor)

        # Continue the search starting at the next node in the prefix tree and the next line in the trace
        tail = match_trace_rec(next_node, log_templates, trace[1:], syntax_tree)

        if tail is None:
            # If the search failed, return none
            return None
        else:
            # If it was successful, prepend the current state
            tail.insert(0, prefix_tree.state)
            return tail
