# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from itertools import chain
from typing import List, Union

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.prefixtree.state import State
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import UnidentifiedLogException

# ****************************************************************************************************
# MatchableGraph
# ****************************************************************************************************

class MatchableGraph(AutoPrinter):
    """
    Abstract class implementing matching methods for a state graph.
    """

    __slots__ = ['syntax_tree', 'start_node']

    def __init__(self, syntax_tree: SyntaxTree, start_node: State):

        assert start_node is not None, "Start node is None!"
        self.syntax_tree = syntax_tree
        self.start_node = start_node

    def get_outgoing_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get outgoing states of a state.

        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """

        pass

    def match_trace(self, trace: List[str], debug: bool = False) -> bool:
        """
        Checks if a given trace matches a path in the graph.
        First, the first line is checked against the start node of the graph.
        If it fits, a successor that matches the next line is picked and the list
        is traversed recursively.

        :param trace: The lines to be matched against the tree.
        :param debug: if True enables printing logs
        :return: If the trace corresponds to a sequence of states in the graph,
                 those states are returned in order. If no match is found, None is returned.
        """

        # If the trace is empty, then it has been fully parsed
        if len(trace) == 0:
            if debug: self.print("Trace is empty")
            return False

        if debug: self.print(f"Matching trace with {len(trace)} lines...")

        # Check which state the current line belongs to
        template = self.syntax_tree.search(trace[0])

        # If template not found, raise exception
        if template is None:
            if debug: self.print(f"Template not found!")
            raise UnidentifiedLogException()

        # Get the root's matching children
        candidates = [state for state in self.get_outgoing_states(self.start_node)
                      if template.name in state.properties.log_templates]

        assert len(candidates) < 2, "Graph is non deterministic!"

        # If no suitable option, return
        if len(candidates) == 0:
            if debug: self.print("Failed matching first line!")
            return False

        # If the trace was only 1 line, check children straight away
        if len(trace) == 1:
            matches = [[1 for state in self.get_outgoing_states(candidate) if state.is_terminal]
                       for candidate in candidates]
            return sum(chain(*matches)) > 0

        # Check paths from every child
        for current_state in candidates:
            if self.__match_trace_rec(current_state, trace[1:]):
                return True

        return False

    def __match_trace(self, state: State, trace: List[str], debug: bool = False) -> bool:
        """
        Iterative helper for the match_trace function.
        This function performs a depth-first search to find a path through the graph
        for the current trace from the current node, if any exists.

        :param state: The current state of the graph
        :param trace: The lines to be matched against the tree.
        :param debug: if True enables printing logs
        :return: True if a path is found, False otherwise.
        """

        if debug: self.print("Matching path from root...")

        paths = []
        curr_state = state
        curr_trace = trace.copy()

        # TODO: check if this works for traces with multiple possible paths
        while len(curr_trace) != 0:

            if debug:
                self.print(f"paths_length={len(paths)}")
                self.print(f"curr_state={curr_state}")
                self.print(f"curr_trace_length={len(curr_trace)}")

            # Find the template of the first line in the syntax tree
            template = self.syntax_tree.search(curr_trace[0])

            # If template not found, raise exception
            if template is None:
                raise UnidentifiedLogException()

            if debug: self.print(f"template={template.name}")

            # Check if any of the children of current node contain the template in their state
            children: List[State] = self.get_outgoing_states(curr_state)
            successors: List[State] = list(
                filter(lambda next_state: template.name in next_state.properties.log_templates, children))

            assert len(successors) < 2, "Graph is non deterministic!"

            if len(successors) == 0:
                # If none found, the trace cannot be matched
                if debug: self.print("\tNo successors found")
                if paths:
                    curr_state, curr_trace = paths.pop(0)
                    continue
                else:
                    break

            if debug: self.print(f"\tFound {len(successors)} successors")

            curr_trace.pop(0)
            curr_state = successors[0]
            if curr_trace:
                if len(successors) > 1:
                    for state in successors[1:]:
                        self.print(f"Added state {state} to paths")
                        paths.append((state, curr_trace.copy()))
            else:
                if debug: self.print(f"Trace finished, curr_state children: {self.get_outgoing_states(curr_state)}")
                matches = [[1 for state in self.get_outgoing_states(candidate) if state.is_terminal]
                           for candidate in successors]
                if sum(chain(*matches)) > 0:
                    return True
                if paths:
                    curr_state, curr_trace = paths.pop(0)

        return False

    def __match_trace_rec(self, state: State, trace: List[str]):
        """
        Recursive helper for the match_trace method.
        This function performs a depth-first recursive search on the tree to find a path through the graph,
        that matches the current trace, if any exists.
        :param state: the current state of the tree
        :param trace: the current trace to match
        :return: True if a path is found that matches the trace, false otherwise
        """

        # If trace finished, check if current state has terminal children
        if not trace:
            matches = [1 for state in self.get_outgoing_states(state) if state.is_terminal]
            return sum(matches) > 0

        # Find the template of the first line in the syntax tree
        template = self.syntax_tree.search(trace[0])
        if template is None:
            raise UnidentifiedLogException()

        # Check if any of the children of current node contain the template in their state
        children: List[State] = self.get_outgoing_states(state)
        successors: List[State] = list(
            filter(lambda next_state: template.name in next_state.properties.log_templates, children))

        assert len(successors) < 2, "Graph is non deterministic!"

        for successor in successors:
            if self.__match_trace_rec(successor, trace[1:]):
                return True

        return False

    def match_templates(self, templates: List[str], debug: bool = False) -> bool:
        """
        Checks if a given list of templates representing a trace has a corresponding path in this graph.
        Assumes that the templates are obtained from the same syntax_tree configuration
        as this graph's instance.
        :param templates: the list of template names to match on
        :param debug: if True enables printing logs to the console
        :return: True if a path was found matching the trace, False otherwise
        """

        # If the trace is empty, then it has been fully parsed
        if len(templates) == 0:
            if debug: self.print("Trace is empty")
            return False

        first_template = templates[0]

        # Get the root's matching children
        candidates = [state for state in self.get_outgoing_states(self.start_node)
                      if first_template in state.properties.log_templates]

        assert len(candidates) < 2, "Graph is non deterministic!"

        # If no suitable option, return
        if len(candidates) == 0:
            if debug: self.print("Failed matching first line!")
            return False

        # If the trace was only 1 line, check children straight away
        if len(templates) == 1:
            matches = [[1 for state in self.get_outgoing_states(candidate) if state.is_terminal]
                       for candidate in candidates]
            return sum(chain(*matches)) > 0

        # Check paths from every child
        for current_state in candidates:
            if self.__match_templates_rec(current_state, templates[1:]):
                return True

        return False

    def __match_templates_rec(self, state: State, templates: List[str]):
        """
        Recursive helper for the match_templates method.
        This function performs a depth-first recursive search on the tree to find a path through the graph,
        that matches the current trace, if any exists.
        :param state: the current state of the tree
        :param templates: the list of templates representing the current trace to match
        :return: True if a path is found that matches the trace, false otherwise
        """

        # If trace finished, check if current state has terminal children
        if not templates:
            matches = [1 for state in self.get_outgoing_states(state) if state.is_terminal]
            return sum(matches) > 0

        template = templates[0]

        # Check if any of the children of current node contain the template in their state
        children: List[State] = self.get_outgoing_states(state)
        successors: List[State] = list(
            filter(lambda next_state: template in next_state.properties.log_templates, children))

        assert len(successors) < 2, "Graph is non deterministic!"

        for successor in successors:
            if self.__match_templates_rec(successor, templates[1:]):
                return True

        return False

    def __len__(self):
        pass
