# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

    __slots__ = ['edges', 'states', 'state_indices_by_id', 'prop_by_hash', 'start_node']

    def __init__(self, start_node: State = None):
        self.edges = SparseMatrix()
        self.states: Dict[int, State] = {}
        self.state_indices_by_id: Dict[int, int] = {}
        self.prop_by_hash: Dict[int, StateProperties] = {}
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

    def remove_loops(self, recurring: bool = False) -> None:
        """
        Main method to remove loops.
        These can be recurring depending on the 'recurring' variable.
        Please view the sub-methods for further explanation
        :param recurring: determines whether recurring loops should be merged, or only singular loops
                          (equivalent subsequent states).
        """
        if recurring:
            self.__remove_recurrent_loops()
        else:
            self.__remove_singular_loops()

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

    def merge_equivalent_children(self, current: State, current_unique: List[State] = None) -> None:
        """
        Merge all equivalent children, such that the resulting automaton remains deterministic while merging.
        :param current: The state of which we want to merge the children
        :param current_unique: The list of states which should remain in the tree, as they are referenced by other
                               nodes.
        """
        if current_unique is None:
            current_unique = []
        children = self.get_outgoing_states_not_self(current)
        children_templates = list(map(lambda x: x.properties.log_templates[0], children))
        duplicates = [i for i, x in enumerate(children_templates) if i != children_templates.index(x)]
        while len(duplicates) > 0:
            for dup in duplicates:
                for c in children:
                    if c.is_equivalent(children[dup]) and c is not children[dup]:
                        if children[dup] in current_unique:
                            self.merge_states(children[dup], c)
                        else:
                            self.merge_states(c, children[dup])
                        break
            children = self.get_outgoing_states_not_self(current)
            if children:
                children_templates = list(map(lambda x: x.properties.log_templates[0], children))
                duplicates = [i for i, x in enumerate(children_templates) if i != children_templates.index(x)]
            else:
                duplicates = []

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
                    self.merge_equivalent_children(current)
                    outgoing = self.get_outgoing_states_not_self(current)

            # Get all outgoing edges except self loops
            outgoing = self.get_outgoing_states_not_self(current)

            # Mark current state as visited
            been.add(current)

            # Continue traversal with all unvisited neighbouring nodes
            for node in [n for n in outgoing if n not in been]:
                stack.append((node, been.copy()))

    def __remove_recurrent_loops(self) -> None:
        """
        Method that removes recurrent loops from branches of a graph.
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
          ||  |
          2   1

          ⊂ indicates a self-loop and 2 has a edge from 1 to 2 and from 2 to 1.
        """

        # Initialize traversal variables
        assert self.start_node is not None
        stack: List[Tuple[State, List]] = [(self.start_node, [])]

        # While there are still unreached nodes
        while len(stack) != 0:

            # Get current state and visited list
            state, current_unique = stack.pop()
            if state not in self:
                continue
            if state in current_unique:  # Should not occur, but does not hurt.
                continue
            else:

                # Get all unvisited neighbours
                outgoing = [x for x in self.get_outgoing_states(state) if x not in current_unique]

                # Get all equivalent states from the unique list
                arr = list(filter(lambda x: state.is_equivalent(x), current_unique))

                if len(arr) > 0:
                    # There will only be one equivalent
                    self.merge_states(arr[0], state)
                    self.merge_equivalent_children(arr[0], current_unique)
                    outgoing = [x for x in self.get_outgoing_states(arr[0]) if x not in current_unique]
                else:
                    # If none are equivalent, mark the current state as unique
                    current_unique.append(state)

                # Continue the traversal for all neighbours of current node
                for s in outgoing:
                    stack.append((s, current_unique.copy()))

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

        self.edges.change_parent_of_children(self.state_indices_by_id[id(state1)],
                                             self.state_indices_by_id[id(state2)])

        self.edges.change_children_of_parents(self.state_indices_by_id[id(state2)],
                                              self.state_indices_by_id[id(state1)])

        del self.states[self.state_indices_by_id[id(state2)]]
        del self.state_indices_by_id[id(state2)]
        del state2

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

    def add_edge(self, start: State, end: State, props: EdgeProperties) -> bool:
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
            results = self.edges.get_parents(self.state_indices_by_id[id(state)])
            return [self.states[result] for result in results] if results else []
        else:
            return None

    def __str__(self):
        return str(self.states)

    def __contains__(self, item: State):
        return id(item) in self.state_indices_by_id

    def __len__(self):
        return len(self.states)
