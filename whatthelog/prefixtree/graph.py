# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import random
from copy import deepcopy
from typing import List, Union, Dict, Tuple, Set

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.state import State
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import StateAlreadyExistsException, StateDoesNotExistException


# ****************************************************************************************************
# Graph
# ****************************************************************************************************
from whatthelog.prefixtree.state_properties import StateProperties


class Graph(AutoPrinter):
    """
    Class implementing a graph
    """

    __slots__ = ['edges', 'states', 'prop_by_hash', 'start_node', 'next_id']

    def __init__(self, start_node: State = None):
        self.outgoing_edges: Dict[State, Set[int]] = {}
        self.incoming_edges: Dict[State, Set[int]] = {}
        self.states: Dict[int, State] = {}
        self.prop_by_hash: Dict[int, StateProperties] = {}
        self.start_node = start_node
        self.next_id: int = 0

    def get_state_by_id(self, state_id: int):
        """
        Method to fetch the state object from its hash.
        :param state_id: the hash of the state to fetch
        :return: the state object
        """
        if state_id not in self.states:
            raise StateDoesNotExistException()
        return self.states[state_id]

    def get_outgoing_states_not_self(self, current: State) -> List[State]:
        """
        Retrieves the outgoing states of a given state, excluding itself in case of a self-loop.
        :param current: The state of which we want to retrieve the outgoing states.
        :return: All the outgoing states, excluding itself in case of a self-loop. In this case that the state is not
                 present, we return an empty list
        """
        outgoing: List[State] = self.get_outgoing_states(current)
        if outgoing:
            return list(filter(lambda x: x is not current, outgoing))
        else:
            return []

    def full_merge_states(self, s1: State, s2: State) -> State:
        """
        Fully merges two states and removes non-determinism.
        :param s1: One of the two states to merge.
        :param s2: One of the two states to merge.
        :return: resulting merged state
        """
        if s2 is None or s1 is None:
            return

        # Trivially merge the states
        self.merge_states(s1, s2)

        # Remove non-determinism in the merged state's children by merging them.
        new_state = self.determinize(s1)

        return new_state

    def merge_states(self, state1: State, state2: State, or_merge = True) -> None:
        """
        This method merges two states and passes the properties from one state to the other.
        :param state1: The new 'merged' state
        :param state2: The state that will be deleted and which properties will be passed to state 1
        """

        props = []

        if or_merge:
            props = state1.properties.log_templates.copy()
            for temp_sequence in state2.properties.log_templates:
                if temp_sequence not in state1.properties.log_templates:
                    props.append(temp_sequence)
        else:
            for temp_sequence in state2.properties.log_templates:
                for original_sequence in state1.properties.log_templates:
                    new_sequence = deepcopy(original_sequence)
                    new_sequence.extend(deepcopy(temp_sequence))
                    props.append(new_sequence)

        state1.properties = StateProperties(props)

        if state2.is_terminal:
            state1.is_terminal = True

        if state2 is self.start_node:
            self.start_node = state1

        if state1.properties.get_prop_hash() in self.prop_by_hash:
            state1.properties = self.prop_by_hash[
                state1.properties.get_prop_hash()]
        else:
            self.prop_by_hash[
                state1.properties.get_prop_hash()] = state1.properties

        self.change_parent_of_children(
            state2, state1)

        self.change_children_of_parents(
            state2, state1)

        del self.states[state2.state_id]
        del state2

    def change_parent_of_children(self, old_parent: State, new_parent: State) -> None:

        # Get the children that need their parent changed
        children = self.get_outgoing_states(old_parent)

        for child in children:
            if child not in self.incoming_edges:
                self.incoming_edges[child] = {new_parent.state_id}
            else:
                self.incoming_edges[child].remove(old_parent.state_id)
                self.incoming_edges[child].add(new_parent.state_id)
            if new_parent not in self.outgoing_edges:
                self.outgoing_edges[new_parent] = {child.state_id}
            else:
                self.outgoing_edges[new_parent].add(child.state_id)

        if old_parent in self.outgoing_edges:
            del self.outgoing_edges[old_parent]

    def change_children_of_parents(self, old_child: State, new_child: State) -> None:
        parents = self.get_incoming_states(old_child)
        for parent in parents:
            if new_child not in self.incoming_edges:
                self.incoming_edges[new_child] = {parent.state_id}
            else:
                self.incoming_edges[new_child].add(parent.state_id)
            if parent not in self.outgoing_edges:
                self.outgoing_edges[parent] = {new_child.state_id}
            else:
                self.outgoing_edges[parent].remove(old_child.state_id)
                self.outgoing_edges[parent].add(new_child.state_id)

        if old_child in self.incoming_edges:
            del self.incoming_edges[old_child]

    def determinize(self, state: State) -> State:
        new_state, changed = self.merge_equivalent_children(state)

        # Get the current state's parent
        parents = self.get_incoming_states(new_state)

        # For each parent
        # TODO: Dont include self in parents in case of self loops
        while len(parents) > 0:
            # Remove non-determinism in the parent
            current, changed = self.merge_equivalent_children(parents.pop())

            # If a change occurred, update the list of parents
            if changed:
                new_state = current
                parents = self.get_incoming_states(current)

        return new_state

    def merge_equivalent_children(self, current: State) -> Tuple[State, bool]:
        """
        Merge all equivalent children, such that the resulting automaton remains deterministic while merging.
        :param current: The state of which we want to merge the children
        """
        merged: bool = False

        # Get all the children of the current node
        # TODO: Refactor duplicate lines
        children = self.get_outgoing_states(current)

        # Get the log templates
        children_templates: List[List[str]] = list(
            map(lambda x: x.properties.log_templates, children))

        # Get a list of duplicate states
        # Two states are duplicates if they have any template in common
        duplicates = []
        for i, x in enumerate(children_templates):
            j = self.__equivalence_index(children_templates, x)
            if i != j:
                duplicates.append((children[i], children[j]))
        has_nondeterminism = len(duplicates) > 0

        # While there are still duplicates left
        while len(duplicates) > 0:

            s1, s2 = duplicates.pop()

            if s1 is current:
                current = s2
            self.merge_states(s2, s1)
            merged = True

            # Update the children and duplicates list
            children = self.get_outgoing_states(current)
            children_templates: List[List[str]] = list(
                map(lambda x: x.properties.log_templates, children))

            # TODO Make this more time efficient ie dont recalculate children but add new ones
            duplicates = []
            for i, x in enumerate(children_templates):
                j = self.__equivalence_index(children_templates, x)
                if i != j:
                    duplicates.append((children[i], children[j]))

        if has_nondeterminism:
            children = self.get_outgoing_states_not_self(current)
            for child in children:
                self.merge_equivalent_children(child)

        return current, merged

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

        self.states[state.state_id] = state

        if state.properties.get_prop_hash() in self.prop_by_hash:
            state.properties = self.prop_by_hash[state.properties.get_prop_hash()]
        else:
            self.prop_by_hash[state.properties.get_prop_hash()] = state.properties

    def add_edge(self, start: State, end: State,
                 props: EdgeProperties = EdgeProperties()) -> bool:
        """
        Method to add an edge to the graph

        :param start: Origin state of the edge
        :param end: Destination state of the edge
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """

        if start.state_id not in self.states or\
                end.state_id not in self.states:
            return False

        if start not in self.outgoing_edges:
            self.outgoing_edges[start] = {end.state_id}
        else:
            if end in self.outgoing_edges[start]:
                return False
            else:
                self.outgoing_edges[start].add(end.state_id)

        if end not in self.incoming_edges:
            self.incoming_edges[end] = {start.state_id}
        else:
            if start in self.incoming_edges[end]:
                return False
            self.incoming_edges[end].add(start.state_id)

        return True

    # def merge_equivalent_children(self, current: State) -> Tuple[State, bool]:
    #     """
    #     Merge all equivalent children, such that the resulting automaton remains deterministic while merging.
    #     :param current: The state of which we want to merge the children
    #     """
    #     merged: bool = False
    #
    #     # Get all the children of the current node except possibly itself
    #     children = self.get_outgoing_states(current)
    #
    #     # Get the log templates
    #     children_templates: List[List[str]] = list(
    #         map(lambda x: x.properties.log_templates, children))
    #
    #     # Get a list of duplicate states
    #     # Two states are duplicates if they have any template in common
    #     duplicates = [i for i, x in enumerate(children_templates)
    #                   if i != self.__equivalence_index(children_templates, x)]
    #
    #     has_nondeterminism = len(duplicates) > 0
    #
    #     # While there are still duplicates left
    #     while len(duplicates) > 0:
    #
    #         # For each duplicate
    #         for dup in duplicates:
    #
    #             for c in children:
    #                 # If a child has a common template with the duplicate, merge them
    #                 if c.is_equivalent_weak(children[dup]) and c is not \
    #                         children[dup]:
    #                     if children[dup] is current:
    #                         current = c
    #                     self.merge_states(c, children[dup])
    #                     merged = True
    #                     break
    #
    #         # Update the children and duplicates list
    #         children = self.get_outgoing_states(current)
    #         if children:
    #             children_templates = list(
    #                 map(lambda x: x.properties.log_templates, children))
    #             duplicates = [i for i, x in enumerate(children_templates)
    #                           if
    #                           i != self.__equivalence_index(children_templates,
    #                                                         x)]
    #         else:
    #             duplicates = []
    #
    #     if has_nondeterminism:
    #         children = self.get_outgoing_states_not_self(current)
    #         for child in children:
    #             self.merge_equivalent_children(child)
    #
    #     return current, merged

    def size(self):
        """
        Method to get the size of the graph.

        :return: Number of states
        """
        return len(self.states)

    def get_outgoing_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get outgoing states of a state.

        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            if state not in self.outgoing_edges:
                return []
            else:
                sids: List[int] = list(self.outgoing_edges[state])
                states: List[State] = []
                for sid in sids:
                    if sid not in self.states:
                        print("Error encountered here...")
                    states.append(self.states[sid])
                return states
        else:
            raise StateDoesNotExistException()

    def get_incoming_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get outgoing states of a state.
        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist raises StateDoesNotExistException.
        """
        if state in self:
            if state not in self.incoming_edges:
                return []
            else:
                return list(map(lambda state_id: self.states[state_id],
                                list(self.incoming_edges[state])))
        else:
            raise StateDoesNotExistException()

    def get_incoming_states_not_self(self, state: State):
        """
        Method to get outgoing states of a state.
        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist raises StateDoesNotExistException.
        """
        incoming = self.get_incoming_states(state)
        if incoming:
            return list(filter(lambda x: x is not state, incoming))
        else:
            return []

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
        return item.state_id in self.states

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

    def matches(self, tree):
        state_graph: State = self.start_node
        state_tree: State = tree.get_outgoing_states_not_self(tree.start_node)[0]

        while not state_tree.is_terminal:
            matching_children = list(filter(lambda state: state.is_equivalent_weak(state_tree),
                                            self.get_outgoing_states(state_graph)))
            if any(matching_children):
                state_graph = matching_children[0]
                state_tree = tree.get_outgoing_states_not_self(state_tree)[0]
            else:
                return False

        return any(filter(lambda state: state.is_equivalent_weak(state_tree),
                          self.get_outgoing_states(state_graph)))

    def matches_unique_states(self, tree):
        # Only works under the assumption that states in this graph are unique!
        state_graph: State = self.start_node
        state_tree: State = tree.get_outgoing_states_not_self(tree.start_node)[0]

        while not state_tree.is_terminal:
            # Get the current template
            current_template: str = state_tree.get_properties().log_templates[0][0]
            # Find matching children in the graph
            found, state_graph, template_sequence = self.matching_children(state_graph, current_template)

            # If there is no child, return false
            if not found:
                return False

            # While the current sequence is not empty
            while template_sequence:
                # If the next element in the sequence does not match the current tree node
                current_elem_in_sequence: str = template_sequence.pop(0)
                prefix_tree_elem: str = state_tree.properties.log_templates[0][0]
                if current_elem_in_sequence != prefix_tree_elem:
                    return False
                # Get the next tree node
                state_tree: State = tree.get_outgoing_states_not_self(state_tree)[0]

                # If it is a terminal node, check if the sequence is empty and if there is a terminal child
                if state_tree.is_terminal:
                    return len(template_sequence) == 0 and self.has_terminal_child(state_graph)

        return False

    def matching_children(self, state: State, template: str) -> Tuple[bool, State, List[str]]:
        for child in self.get_outgoing_states(state):
            for template_sequence in child.get_properties().log_templates:
                if template_sequence[0] == template:
                    return True, child, deepcopy(template_sequence)

        return False, State([]), []

    def has_terminal_child(self, state: State):
        return any(map(lambda s: s.is_terminal, self.get_outgoing_states(state)))

    def get_non_terminal_state_and_child(self) -> Union[Tuple[State, State], None]:
        if len(self.states) <= 3:
            return None

        state: State = self.get_random_state()
        children: List[State] = self.get_outgoing_states_not_self(state)
        while state.is_terminal or (len(children) == 1 and children[0].is_terminal) \
                or (len(children) == 0) or state is self.start_node:
            state = self.get_random_state()
            children: List[State] = self.get_outgoing_states_not_self(state)

        child = self.get_random_child(state)
        while child.is_terminal or child is state:
            child = self.get_random_child(state)

        return state, child

    def get_non_terminal_states(self) -> Tuple[State, State]:
        state: State = self.get_random_state()
        children: List[State] = self.get_outgoing_states_not_self(state)
        while state.is_terminal or (len(children) == 1 and children[0].is_terminal) \
                or (len(children) == 0) or state is self.start_node:
            state = self.get_random_state()
            children: List[State] = self.get_outgoing_states_not_self(state)

        other_state: State = self.get_random_state()
        other_children: List[State] = self.get_outgoing_states_not_self(other_state)
        while state.is_terminal or (len(other_children) == 1 and other_children[0].is_terminal) \
                or (len(other_children) == 0) or other_state is self.start_node\
                or other_state is state:
            other_state = self.get_random_state()
            other_children: List[State] = self.get_outgoing_states_not_self(other_state)

        return state, other_state
