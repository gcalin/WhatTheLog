import random
from typing import List, Union, Dict

from whatthelog.prefixtree.prefix_tree import PrefixTree, State
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


def template_matches_state(template: str, state: State) -> bool:
    return template in state.properties.log_templates


def match_trace(
        prefix_tree: PrefixTree,
        trace: List[str],
        syntax_tree: SyntaxTree) -> Union[List[State], None]:
    """
    Checks if a given trace matches a path in the prefix tree.
    First, the first line is checked against the node of the prefix tree.
    If it fits, a successor that matches the next line is picked and the list
    is traversed recursively.

    :param prefix_tree: The prefix tree in which states need to be matched.
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

    # If no state is found, return None
    if template is None:
        return None

    # Get the root's children
    root_children = list(
        filter(lambda s: template_matches_state(template.name, s),
               prefix_tree.get_children(prefix_tree.get_root())))

    # If no suitable option, return
    if len(root_children) == 0:
        return None

    # Randomly pick first suitable state
    current_state: State = random.choice(root_children)

    # Check if the template matches the root of the state tree
    if template_matches_state(template.name, current_state):

        # If the trace is exactly one line long, return the root state
        if len(trace) == 1:
            return [current_state]

        # Remove the checked line from trace
        trace[:] = trace[1:]

        # Find the state for the second line
        template = syntax_tree.search(trace[0])

        # If no state is found, raise an exception
        if template is None:
            return None

        # Check if any of the children of current node contain the template in their state
        children: List[State] = prefix_tree.get_children(current_state)
        successor: List[State] = list(filter(lambda next_state: template_matches_state(template.name, next_state), children))

        if len(successor) == 0:
            # If none found, the trace cannot be matched
            return None
        else:
            # Pick a random suitable next node
            # TODO: Should this be removed through an invariant in the prefix tree?
            next_node: State = random.choice(successor)

            # Continue the search recursively
            tail: List[State] = match_trace_rec(next_node, prefix_tree, trace[1:], syntax_tree)

            if tail is None:
                # If the search failed, return none
                return None
            else:
                # If it was successful, prepend the current state
                tail.insert(0, current_state)
                return tail


def match_trace_rec(
        current_state: State,
        prefix_tree: PrefixTree,
        trace: List[str],
        syntax_tree: SyntaxTree) -> Union[List[State], None]:
    """
    Recursive helper for the match_trace function. This function recursively
    finds the next state in the prefix tree for a the first line in the trace
    if any exists.

    :param current_state: The current state of the prefix tree
    :param prefix_tree: The prefix tree in which states need to be matched.
    :param trace: The lines to be matched against the tree.
    :param syntax_tree: The syntax tree used to validate the lines.
    :return: If the trace corresponds to a sequence of states in the prefix tree,
             those states are returned in order. If no match is found, None is returned.
    """
    res = [current_state]

    while len(trace) != 0:
        # Find the template of the first line in the syntax tree
        template = syntax_tree.search(trace[0])

        # If no state is found, raise an exception
        if template is None:
            return None

        # Check if any of the children of current node contain the template in their state
        children: List[State] = prefix_tree.get_children(current_state)
        successor: List[State] = list(
            filter(lambda next_state: template_matches_state(template.name, next_state), children))

        if len(successor) == 0:
            # If none found, the trace cannot be matched
            return None
        else:
            # Pick a random suitable next node
            current_state = random.choice(successor)

            # Append result to the list of states
            res.append(current_state)

            # Remove first trace
            trace = trace[1:]

    if [state.is_terminal for state in prefix_tree.get_children(current_state) if state.is_terminal]:
        return res
    return None
