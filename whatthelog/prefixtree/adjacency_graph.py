from collections import Iterable
from typing import List, Union, Dict, Tuple

from whatthelog.exceptions import StateAlreadyExistsException, \
    StateDoesNotExistException, InvalidEdgeException
from whatthelog.prefixtree.matchable_graph import MatchableGraph
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.state import State
from whatthelog.prefixtree.state_properties import StateProperties
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class AdjacencyGraph(MatchableGraph, Iterable):
    """
    Class implementing a graph
    """

    __slots__ = ['syntax_tree', 'states', 'state_indices_by_id', 'prop_by_hash',
                 'start_node', 'outgoing_edges', 'incoming_edges']

    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state):

        for slot in state:
            setattr(self, slot, state[slot])

        # --- Rebuild state indices table ---
        self.state_indices_by_id = {}
        for index, s in self.states.items():
            self.state_indices_by_id[id(s)] = index

    def __init__(self, syntax_tree: SyntaxTree, start_node: State = None):

        super().__init__(syntax_tree, start_node)
        self.states: Dict[int, State] = {}
        self.state_indices_by_id: Dict[int, int] = {}
        self.outgoing_edges: Dict[State, Dict[State, EdgeProperties]] = {}
        self.incoming_edges: Dict[State, Dict[State, EdgeProperties]] = {}
        self.prop_by_hash: Dict[int, StateProperties] = {}
        if start_node is not None:
            self.add_state(start_node)

    def get_state_by_id(self, state_id: int):
        """
        Method to fetch the state object from its hash.
        :param state_id: the hash of the state to fetch
        :return: the state object
        """
        if state_id not in self.state_indices_by_id:
            raise StateDoesNotExistException()
        return self.states[self.state_indices_by_id[state_id]]

    def get_state_index_by_id(self, state_id: int):
        """
        Method to fetch the state object from its hash.
        :param state_id: the hash of the state to fetch
        :return: the state object
        """
        if state_id not in self.state_indices_by_id:
            raise StateDoesNotExistException()
        return self.state_indices_by_id[state_id]

    def get_outgoing_props(self, state: State) -> Union[List[EdgeProperties], None]:
        """
        Method to get outgoing edges of a state.

        :param state: State to get outgoing edges for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            results = self.get_outgoing_states(state)
            return [EdgeProperties.parse(result.properties) for result in results]
        else:
            return None

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
            state.properties = self.prop_by_hash[
                state.properties.get_prop_hash()]
        else:
            self.prop_by_hash[
                state.properties.get_prop_hash()] = state.properties

    def add_edge(self, start: State, end: State,
                 props: EdgeProperties = EdgeProperties()) -> bool:
        """
        Method to add an edge to the graph
        :param start: Origin state of the edge
        :param end: Destination state of the edge
        :param props: the edge properties object
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """

        if id(start) not in self.state_indices_by_id or id(
                end) not in self.state_indices_by_id:
            return False

        if start not in self.outgoing_edges:
            self.outgoing_edges[start] = {end: props}
        else:
            if end in self.outgoing_edges[start]:
                return False
            else:
                self.outgoing_edges[start][end] = props

        if end not in self.incoming_edges:
            self.incoming_edges[end] = {start: props}
        else:
            if start in self.incoming_edges[end]:
                return False
            self.incoming_edges[end][start] = props

        return True

    def get_outgoing_states(self, state: State) -> List[State]:
        """
        Method to get outgoing states of a state.
        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist raises StateDoesNotExistException.
        """
        if state in self:
            if state not in self.outgoing_edges:
                return []
            else:
                return list(self.outgoing_edges[state].keys())
        else:
            raise StateDoesNotExistException()

    def get_outgoing_states_not_self(self, current: State) -> List[State]:
        """
        Retrieves the outgoing states of a given state, excluding itself in case of a self-loop (extra O(n)).
        :param current: The state of which we want to retrieve the outgoing states.
        :return: All the outgoing states, excluding itself in case of a self-loop. In this case that the state is not
                 present, we return an empty list
        """
        outgoing = self.get_outgoing_states(current)
        if outgoing:
            return list(filter(lambda x: x is not current, outgoing))
        else:
            return []

    def get_incoming_states(self, state: State):
        """
        Method to get incoming states of a state.
        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist raises StateDoesNotExistException.
        """
        if state in self:
            if state not in self.incoming_edges:
                return []
            else:
                return list(self.incoming_edges[state].keys())
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

    def update_edge(self, start: State, end: State, passes: int = 1):
        """
        Method to update an edge by adding a given number of passes (default is 1).
        :param start: the start node of the edge
        :param end: the end node of the edge
        :param passes: the number of passes to update the edge with (default is 1)
        :return: True if the update succeeded, else otherwise
        """

        if start not in self or end not in self:
            return False

        if start not in self.outgoing_edges or end not in self.incoming_edges:
            raise InvalidEdgeException()
        elif end not in self.outgoing_edges[start] or start not in self.incoming_edges[end]:
            raise InvalidEdgeException()
        else:
            self.outgoing_edges[start][end].passes += passes

    def full_merge_states(self, s1: State, s2: State):
        """
        Fully merges two states and removes non-determinism.
        :param s1: One of the two states to merge.
        :param s2: One of the two states to merge.
        :return: resulting merged state
        """

        if s2 is None or s1 is None:
            return None

        # Trivially merge the states
        self.merge_states(s1, s2)

        # Recursively remove non-determinism around the merged state.
        self.determinize(s1)
        # for parent in self.get_incoming_states_not_self(s1):
        #     if parent in self:
        #         self.determinize(parent)

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
            state1.properties = self.prop_by_hash[
                state1.properties.get_prop_hash()]
        else:
            self.prop_by_hash[
                state1.properties.get_prop_hash()] = state1.properties

        self.change_parent_of_children(
            state2, state1)

        self.change_child_of_parents(
            state2, state1)

        del self.states[self.state_indices_by_id[id(state2)]]
        del self.state_indices_by_id[id(state2)]
        del state2

    def determinize(self, state: State):
        """
        Checks the given state's children for non-determinism.
        If any duplicates are found, merges them
        and recursively checks their children for non-determinism.
        :param state: the state to determinize
        :return:
        """

        assert state in self, "State not in self"
        duplicates = self.__get_equivalent_states(self.get_outgoing_states_not_self(state))
        while duplicates:

            s1, s2 = duplicates
            self.full_merge_states(s1, s2)

            if state not in self: break
            duplicates = self.__get_equivalent_states(self.get_outgoing_states_not_self(state))

    def change_parent_of_children(self, old_parent: State, new_parent: State) -> None:
        children = self.get_outgoing_states(old_parent)
        for child in children:
            old_edge = self.incoming_edges[child].pop(old_parent)
            if child not in self.incoming_edges:
                self.incoming_edges[child] = {new_parent: old_edge}
            else:
                self.incoming_edges[child][new_parent] = old_edge
            if new_parent not in self.outgoing_edges:
                self.outgoing_edges[new_parent] = {child: old_edge}
            else:
                self.outgoing_edges[new_parent][child] = old_edge

        if old_parent in self.outgoing_edges:
            del self.outgoing_edges[old_parent]

    def change_child_of_parents(self, old_child: State, new_child: State) -> None:
        parents = self.get_incoming_states(old_child)
        for parent in parents:
            old_edge = self.outgoing_edges[parent].pop(old_child)
            if new_child not in self.incoming_edges:
                self.incoming_edges[new_child] = {parent: old_edge}
            else:
                self.incoming_edges[new_child][parent] = old_edge
            if parent not in self.outgoing_edges:
                self.outgoing_edges[parent] = {new_child: old_edge}
            else:
                self.outgoing_edges[parent][new_child] = old_edge

        if old_child in self.incoming_edges:
            del self.incoming_edges[old_child]

    def merge_equivalent_children(self, current: State):
        """
        Merge all equivalent children, such that the resulting automaton remains deterministic while merging.
        :param current: The state of which we want to merge the children
        """

        # Get the log templates
        children = self.get_outgoing_states_not_self(current)
        children_templates: List[List[str]] = [x.properties.log_templates for x in children]

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
                if child in self:
                    self.merge_equivalent_children(child)

    def get_adj_list(self, remove_self_loops: bool = False, as_strings: bool = False) \
            -> Union[List[Tuple[int, int, float]], List[str]]:
        """
        Returns an adjacency list representing the prefix tree, as a list of tuples in the form:
            (edge_start, edge_end, edge_weight).

        Each edge's weight is the frequency with which that edge is used from its parent,
        in other words: if node A has two children B and C, with edge AB having 6 passes
        and edge AC having 4 passes, then edge AB will have weight 0.6 andAC 0.4.

        :param remove_self_loops: if True the output will not contain self-looping edges
                                  (edges with same origin as destination)
        :param as_strings: if True returns the edges as strings separated by a whitespace
        :return: the adjacency matrix for the prefix tree
        """

        adjacency = []
        for state, children in self.outgoing_edges.items():

            if state == self.start_node: continue

            state_idx = self.state_indices_by_id[id(state)]
            edges = []
            for child, props in children.items():

                child_idx = self.state_indices_by_id[id(child)]
                if remove_self_loops and child_idx == state_idx: continue

                edges.append((child_idx, props.passes))

            total = sum(i for _, i in edges)

            for child_idx, passes in edges:
                if as_strings:
                    adjacency.append(f"{state_idx} {child_idx} {1 - (passes/total)}")
                else:
                    adjacency.append((state_idx, child_idx, 1 - (passes/total)))

        return adjacency

    def __remove_singular_loops(self) -> None:
        """
        Removes equivalent subsequent loops from the graph.
        For example if graph is:
            0
           /\
          1  3
          1  3
          1  1
          2
          1
        The output will be:
            0
           / \
          ⊂1 ⊂3
          |   |
          2   1
          |
          1
          ⊂ indicates a self-loop.
        """

        # Initialize traversal variables
        assert self.start_node is not None
        current = self.start_node
        been = set()
        stack = [(current, been)]

        # While there are still unreached nodes
        while len(stack) > 0:
            # Get the first node and its neighbours
            current, been = stack.pop()
            outgoing = self.get_outgoing_states_not_self(current)

            # Merge all states directly linked to and equivalent to the current state
            while len(outgoing) > 0:
                out = outgoing.pop()
                if current.is_equivalent(out) and current is not out:
                    self.merge_states(current, out)
                    outgoing = self.get_outgoing_states_not_self(current)

            self.merge_equivalent_children(current)

            # Get all outgoing edges except self loops
            outgoing = self.get_outgoing_states_not_self(current)

            # Mark current state as visited
            been.add(current)

            # Continue traversal with all unvisited neighbouring nodes
            for node in [n for n in outgoing if n not in been]:
                stack.append((node, been.copy()))

    @staticmethod
    def __get_equivalent_states(target_states: List[State]) -> Union[Tuple[State, State], None]:
        """
        Find equivalent states in a given list, or None if none found.
        Equivalent states are defined as having at least one log template in common.
        :param target_states: the list of states to search for equivalents.
        :return: A tuple of equivalent states if any found, None otherwise.
        """

        generator = ((x, y) for x in target_states for y in target_states if x != y)
        for x, y in generator:
            if set(x.properties.log_templates) & set(y.properties.log_templates):
                return x, y

    @staticmethod
    def __equivalence_index(target_list: List[List[str]],
                            target_items: List[str]) -> Union[int, None]:
        """
        For a given template, find the first index within a list that has a weakly equivalent template.
        Weak equivalence implies that at least one template is common.
        """
        for i, l in enumerate(target_list):
            for item in target_items:
                if item in l:
                    return i
        return None

    def __str__(self):
        return str(self.states)

    def __contains__(self, item: Union[State, int]):
        if isinstance(item, State):
            return id(item) in self.state_indices_by_id
        elif isinstance(item, int):
            return item in self.states
        raise TypeError(f"Contains expects a State or int but {type(item)} was given!")

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return TreeIterator(self)

#****************************************************************************************************
# Prefix Tree Iterator
#****************************************************************************************************

class TreeIterator:
    """
    Iterator class for the tree.
    Return states in a Breadth-First Search.
    """

    def __init__(self, graph: AdjacencyGraph):
        self.graph = graph
        self.indices = list(graph.states.keys())
        self.current = self.indices.pop(0)

    def __next__(self) -> State:

        if not self.indices:
            raise StopIteration

        return self.graph.states[self.indices.pop(0)]