# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from copy import deepcopy
from itertools import chain
import random
from typing import List, Union, Dict, Tuple

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.prefixtree.state import State
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import StateAlreadyExistsException, StateDoesNotExistException, UnidentifiedLogException
from whatthelog.prefixtree.sparse_matrix import SparseMatrix
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.state_properties import StateProperties


# ****************************************************************************************************
# Graph
# ****************************************************************************************************

class Graph(AutoPrinter):
    """
    Class implementing a graph
    """

    __slots__ = ['syntax_tree', 'edges', 'states', 'state_indices_by_id', 'prop_by_hash', 'start_node']

    def __init__(self, syntax_tree: SyntaxTree, start_node: State):

        assert start_node is not None, "Start node is None!"
        self.syntax_tree = syntax_tree
        self.edges = SparseMatrix()
        self.states: Dict[int, State] = { 0: start_node }
        self.state_indices_by_id: Dict[int, int] = { id(start_node): 0 }
        self.prop_by_hash: Dict[int, StateProperties] = { hash(start_node.properties): start_node.properties }
        self.start_node = start_node

    def get_state_by_id(self, state_id: int):
        """
        Method to fetch the state object from its hash.
        :param state_id: the hash of the state to fetch
        :return: the state object
        """
        if state_id not in self.state_indices_by_id:
            raise StateDoesNotExistException()
        return self.states[self.state_indices_by_id[state_id]]

    def get_outgoing_states_not_self(self, current: State) -> List[State]:
        """
        Retrieves the outgoing states of a given state, excluding itself in case of a self-loop.
        :param current: The state of which we want to retrieve the outgoing states.
        :return: All the outgoing states, excluding itself in case of a self-loop. In this case that the state is not
                 present, we return an empty list
        """
        outgoing = self.get_outgoing_states(current)
        if outgoing:
            return list(filter(lambda x: x is not current, outgoing))
        else:
            return []

    def merge_states(self, state1: State, state2: State) -> None:
        """
        This method merges two states and passes the properties from one state to the other.
        :param state1: The new 'merged' state
        :param state2: The state that will be deleted and which properties will be passed to state 1
        """

        if state1 is None or state2 is None:
            return

        assert state1 in self and state2 in self, "State not in graph!"

        props = state1.properties.log_templates.copy()

        for temp in state2.properties.log_templates:
            if temp not in state1.properties.log_templates:
                props.append(temp)
        state1.properties = StateProperties(props)

        if state2.is_terminal:
            state1.is_terminal = True

        if state2 is self.start_node:
            self.start_node = state1

        if state1.properties.get_prop_hash() in self.prop_by_hash:
            state1.properties = self.prop_by_hash[state1.properties.get_prop_hash()]
        else:
            self.prop_by_hash[state1.properties.get_prop_hash()] = state1.properties

        self.edges.change_parent_of_children(self.state_indices_by_id[id(state1)],
                                             self.state_indices_by_id[id(state2)])

        self.edges.change_children_of_parents(self.state_indices_by_id[id(state2)],
                                              self.state_indices_by_id[id(state1)])

        del self.states[self.state_indices_by_id[id(state2)]]
        del self.state_indices_by_id[id(state2)]
        del state2

    def full_merge_states(self, s1: State, s2: State):
        """
        Fully merges two states and removes non-determinism.
        :param s1: One of the two states to merge.
        :param s2: One of the two states to merge.
        """

        # Trivially merge the states
        self.merge_states(s1, s2)

        # Remove non-determinism in the merged state's children by merging them.
        current, changed = self.merge_equivalent_children(s1)

        # Get the current state's parent
        parents = self.get_incoming_states(current)

        # For each parent
        while len(parents) > 0:
            # Remove non-determinism in the parent
            current, changed = self.merge_equivalent_children(parents.pop())

            # If a change occurred, update the list of parents
            if changed:
                current, changed = self.merge_equivalent_children(current)
                parents = self.get_incoming_states(current)

    def add_state(self, state: State) -> None:
        """
        Method to add a new state to the graph.

        :param state: State to add
        :raises StateAlreadyExistsException: if state already exists
        """

        if state in self:
            raise StateAlreadyExistsException()

        curr_index = len(self.states)
        self.states[curr_index] = state
        self.state_indices_by_id[id(state)] = curr_index

        if state.properties.get_prop_hash() in self.prop_by_hash:
            state.properties = self.prop_by_hash[state.properties.get_prop_hash()]
        else:
            self.prop_by_hash[state.properties.get_prop_hash()] = state.properties

    def add_edge(self, start: State, end: State, props: EdgeProperties = EdgeProperties()) -> bool:
        """
        Method to add an edge to the graph

        :param start: Origin state of the edge
        :param end: Destination state of the edge
        :param props: the edge properties object
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """

        if id(start) not in self.state_indices_by_id or id(end) not in self.state_indices_by_id:
            return False

        start_index = self.state_indices_by_id[id(start)]
        end_index = self.state_indices_by_id[id(end)]
        if not (start_index, end_index) in self.edges:

            self.edges[start_index, end_index] = str(props)
            return True
        return False

    def merge_equivalent_children(self, current: State) -> Tuple[State, bool]:
        """
        Merge all equivalent children, such that the resulting automaton remains deterministic while merging.
        :param current: The state of which we want to merge the children
        """

        merged: bool = False

        # Get all the children of the current node except possibly itself
        children = self.get_outgoing_states(current)

        # Get the log templates
        children_templates: List[List[str]] = list(
            map(lambda x: x.properties.log_templates, children))

        # Get a list of duplicate states
        # Two states are duplicates if they have any template in common
        duplicates = [i for i, x in enumerate(children_templates)
                      if i != self.__equivalence_index(children_templates, x)]

        has_nondeterminism = len(duplicates) > 0

        # While there are still duplicates left
        while len(duplicates) > 0:

            # For each duplicate
            for dup in duplicates:

                for c in children:
                    # If a child has a common template with the duplicate, merge them
                    if c.is_equivalent_weak(children[dup]) and c is not \
                            children[dup]:
                        if children[dup] is current:
                            current = c
                        self.merge_states(c, children[dup])
                        merged = True
                        break

            # Update the children and duplicates list
            children = self.get_outgoing_states(current)
            if children:
                children_templates = list(
                    map(lambda x: x.properties.log_templates, children))
                duplicates = [i for i, x in enumerate(children_templates)
                              if
                              i != self.__equivalence_index(children_templates,
                                                            x)]
            else:
                duplicates = []

        if has_nondeterminism:
            children = self.get_outgoing_states_not_self(current)
            for child in children:
                self.merge_equivalent_children(child)

        return current, merged

    def get_outgoing_props(self, state: State) -> Union[List[EdgeProperties], None]:
        """
        Method to get outgoing edges of a state.

        :param state: State to get outgoing edges for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            results = self.edges.find_children(self.state_indices_by_id[id(state)])
            return [EdgeProperties.parse(result[1]) for result in results]
        else:
            return None

    def get_outgoing_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get outgoing states of a state.

        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """

        if state in self:
            results = self.edges.find_children(self.state_indices_by_id[id(state)])
            return [self.states[result[0]] for result in results] if results else []
        else:
            return None

    def get_incoming_states(self, state: State):
        """
                Method to get outgoing states of a state.

                :param state: State to get outgoing states for
                :return: List of outgoing edges from state.
                If state does not exist return None.
                """
        if state in self:
            results = self.edges.get_parents(self.state_indices_by_id[id(state)])
            return [self.states[result] for result in results] if results else []
        else:
            return None

    def update_edge(self, start: State, end: State, passes: int = 1):
        """
        Method to update an edge by adding a given number of passes (default is 1).
        :param start: the start node of the edge
        :param end: the end node of the edge
        :param passes: the number of passes to update the edge with (default is 1)
        :return: True if the update succeeded, else otherwise
        """

        if id(start) not in self.state_indices_by_id or id(end) not in self.state_indices_by_id:
            return False

        start_index = self.state_indices_by_id[id(start)]
        end_index = self.state_indices_by_id[id(end)]
        if (start_index, end_index) in self.edges:
            self.edges[start_index, end_index] = int(self.edges[start_index, end_index]) + passes
            return True
        return False

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
            if self.__match_trace(current_state, trace[1:], debug):
                return True

        return False

    def __match_trace(self, state: State, trace: List[str], debug: bool = False) -> bool:
        """
        Recursive helper for the match_trace function. This function recursively
        finds the next state in the graph for the first line in the trace
        if any exists.

        :param state: The current state of the graph
        :param trace: The lines to be matched against the tree.
        :param debug: if True enables printing logs
        :return: If the trace corresponds to a sequence of states in the prefix tree,
                 those states are returned in order. If no match is found, None is returned.
        """

        if debug: self.print("Matching path from root...")

        paths = []
        curr_state = state
        curr_trace = trace.copy()

        # TODO: check if this works for traces with multiple possible paths
        while len(curr_trace) != 0:

            if debug:
                self.print(f"paths={paths}")
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
                        paths.append((state, curr_trace.copy()))
            else:
                if debug: self.print(f"Trace finished, curr_state children: {self.get_outgoing_states(curr_state)}")
                if [state.is_terminal for state in self.get_outgoing_states(curr_state) if state.is_terminal]:
                    return True
                if paths:
                    curr_state, curr_trace = paths.pop(0)

        return False

    def get_random_state(self) -> State:
        """
        Gets a random state in the graph.
        """
        return random.choice(list(self.states.values()))

    def get_random_child(self, state: State) -> State:
        """
        Gets a random child of a given state.
        """
        return random.choice(self.get_outgoing_states_not_self(state))

    def __str__(self):
        return str(self.states)

    def __contains__(self, item: State):
        return id(item) in self.state_indices_by_id

    def __len__(self):
        return len(self.states)

    def __getstate__(self):
        return { slot: getattr(self, slot) for slot in self.__slots__ }

    def __setstate__(self, state):

        for slot in state:
            setattr(self, slot, state[slot])

        # --- Rebuild state indices table ---
        self.state_indices_by_id = {}
        for index, state in self.states.items():
            self.state_indices_by_id[id(state)] = index

    def __deepcopy__(self, memodict={}) -> Graph:

        edges = deepcopy(self.edges)
        states = deepcopy(self.states)
        prop_by_hash = deepcopy(self.prop_by_hash)

        start_index = self.state_indices_by_id[id(self.start_node)]
        new_start = states[start_index]

        output = Graph(deepcopy(self.syntax_tree), new_start)
        output.edges = edges
        output.states = states
        output.prop_by_hash = prop_by_hash

        # --- Rebuild state indices table ---
        output.state_indices_by_id = {}
        for index, state in output.states.items():
            output.state_indices_by_id[id(state)] = index

        return output

    @staticmethod
    def __equivalence_index(target_list: List[List[str]], target_items: List[str]) -> Union[int, None]:
        """
        For a given template, find the first index within a list that has a weakly equivalent template.
        Weak equivalence implies that at least one template is common.
        """
        for i, l in enumerate(target_list):
            for item in target_items:
                if item in l:
                    return i
        return None
