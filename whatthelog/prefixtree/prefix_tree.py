from typing import List, Union

from whatthelog.prefixtree.state import State


class PrefixTree:
    """
    Class representing a recursive prefix tree data structure with
     each node holding a State and a list of children.
    """
    def __init__(self, state: State, parent: Union['PrefixTree', None]):
        """
        Prefix tree node constructor. Holds the state it represents,
        a reference to its parent and a list of children. If this node is the root,
        the parent is set to None. If it is a leave the children list is empty.

        :param state: The state this node represents
        :param parent: A reference to the parent of this node
        """
        self.state: State = state
        self.__parent: Union['PrefixTree', None] = parent
        self.__children: List['PrefixTree'] = []

    def set_state(self, state: State):
        """
        Method to set the state of the node.

        :param state: State to be set
        """
        self.state = state

    def get_children(self) -> List['PrefixTree']:
        """
        Method to get children of the node.

        :return: This nodes children
        """
        return self.__children

    def get_parent(self) -> 'PrefixTree':
        """
        Method to get the parent of the node.

        :return: This node's parent.
        """
        return self.__parent

    def add_child(self, child: 'PrefixTree'):
        """
        Method to add a child to the node.

        :param child: Child to be added
        """
        self.__children.append(child)

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
