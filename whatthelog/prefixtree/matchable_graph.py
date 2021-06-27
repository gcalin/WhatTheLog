# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
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

    def get_state_index_by_id(self, state_id: int) -> int:
        """
        Method to get the index of a state from its ID.
        Throws a StateDoesNotExistException if state not found.
        :param state_id: the ID of the state to fetch the index of.
        :return: the index of the state.
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

        templates = [self.syntax_tree.search(log) for log in trace]
        if None in templates:
            if debug: self.print(f"Template not found!")
            raise UnidentifiedLogException()
        templates = [tree.name for tree in templates]

        return self.match_templates(templates, debug=debug)

    def match_templates(self, templates: List[str], dfs: bool = True, debug: bool = False) -> bool:
        """
        Checks if a given list of templates representing a trace has a corresponding path in this graph.
        Assumes that the templates are obtained from the same syntax_tree configuration
        as this graph's instance.
        This method supports searching through a nondeterministic graph by following each possible path,
        this however results in an exponential worst-case time complexity
        on the number of non-deterministic transitions in the graph.
        :param templates: the trace to find a path for
        :param dfs: if True a Depth-First Search strategy will be used to search for a path,
                    otherwise a Breadth-First Search one will be used.
        :param debug: if True enables logging to the console
        :return: True if a path is found, False otherwise
        """

        # If the trace is empty, then it has been fully parsed
        if not templates:
            if debug: self.print("Trace is empty")
            return False

        # Get children of root matching first trace
        candidates = [state for state in self.get_outgoing_states(self.start_node)
                      if templates[0] in state.properties.log_templates]
        if not candidates:
            if debug: self.print("Failed matching first trace")
            return False

        # If trace is only 1 template long, check matches directly
        if len(templates) == 1:
            return sum([1 for state in candidates if state.is_terminal]) > 0

        # Search children of first candidates
        queue = [(state, templates[1:]) for state in candidates]
        while queue:

            current_path = queue.pop(0)
            current_state = current_path[0]
            current_trace = current_path[1]

            if debug: self.print(f"Matching trace of {len(current_trace)} logs "
                                 f"from state {self.get_state_index_by_id(id(current_state))}")

            # Get children of current node matching current trace
            candidates = [state for state in self.get_outgoing_states(current_state)
                          if current_trace[0] in state.properties.log_templates]
            if candidates:

                # If trace is finished, check if final states are terminal
                if len(current_trace) == 1:

                    is_terminal = sum([1 for state in candidates if state.is_terminal]) > 0
                    if is_terminal:
                        return True
                else:

                    # Append possible paths to queue
                    for candidate in candidates:
                        if dfs:
                            queue.insert(0, (candidate, current_trace[1:]))
                        else:
                            queue.append((candidate, current_trace[1:]))

        return False

    @staticmethod
    def template_matches_state(template: str,
                               state: State) -> bool:
        return template in state.properties.log_templates

    def __len__(self):
        pass
