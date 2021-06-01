# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import random
from typing import List, Union, Dict, Tuple

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state import State
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import StateAlreadyExistsException, StateDoesNotExistException
from whatthelog.prefixtree.sparse_matrix import SparseMatrix
from whatthelog.prefixtree.edge_properties import EdgeProperties


# ****************************************************************************************************
# Graph
# ****************************************************************************************************
from whatthelog.prefixtree.state_properties import StateProperties


class Graph(AutoPrinter):
    """
    Class implementing a graph
    """

    __slots__ = ['edges', 'states', 'state_indices_by_id', 'prop_by_hash', 'start_node', 'next_id']

    def __init__(self, start_node: State = None):
        self.edges = SparseMatrix()
        self.states: Dict[int, State] = {}
        self.state_indices_by_id: Dict[int, int] = {}
        self.prop_by_hash: Dict[int, StateProperties] = {}
        self.start_node = start_node
        self.next_id: int = 0

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

        self.edges.change_parent_of_children(self.state_indices_by_id[state1.state_id],
                                             self.state_indices_by_id[state2.state_id])

        self.edges.change_children_of_parents(self.state_indices_by_id[state2.state_id],
                                              self.state_indices_by_id[state1.state_id])

        del self.states[self.state_indices_by_id[state2.state_id]]
        del self.state_indices_by_id[state2.state_id]
        del state2

    def full_merge_states(self, s1: State, s2: State):
        """
        Fully merges two states and removes non-determinism.
        :param s1: One of the two states to merge.
        :param s2: One of the two states to merge.
        """
        if s2 is None or s1 is None:
            return

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
        state.state_id = self.next_id
        self.next_id += 1

        if state in self:
            raise StateAlreadyExistsException()

        curr_index = len(self.states)
        self.states[curr_index] = state
        self.state_indices_by_id[state.state_id] = curr_index

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

        if start.state_id not in self.state_indices_by_id or end.state_id not in self.state_indices_by_id:
            return False

        start_index = self.state_indices_by_id[start.state_id]
        end_index = self.state_indices_by_id[end.state_id]
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

    def size(self):
        """
        Method to get the size of the graph.

        :return: Number of states
        """
        return len(self.states)

    def get_outgoing_props(self, state: State) -> Union[List[EdgeProperties], None]:
        """
        Method to get outgoing edges of a state.

        :param state: State to get outgoing edges for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            results = self.edges.find_children(self.state_indices_by_id[state.state_id])
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
            results = self.edges.find_children(self.state_indices_by_id[state.state_id])
            return [self.states[result[0]] for result in results] if results is not None else []
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
            results = self.edges.get_parents(self.state_indices_by_id[state.state_id])
            return [self.states[result] for result in results] if results else []
        else:
            return None

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
        return item.state_id in self.state_indices_by_id

    def __len__(self):
        return len(self.states)

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
