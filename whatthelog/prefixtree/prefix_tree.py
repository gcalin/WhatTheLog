#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from typing import List, Union

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state import State
from whatthelog.auto_printer import AutoPrinter


#****************************************************************************************************
# Syntax Tree
#****************************************************************************************************

class PrefixTree(AutoPrinter):

    """
    Class representing a recursive prefix tree data structure with
    each node holding a State and a list of children.
    """
    def __init__(self, state: State, parent: Union[PrefixTree, None], children: Union[List[PrefixTree], None] = None):
        """
        Prefix tree node constructor. Holds the state it represents,
        a reference to its parent and a list of children. If this node is the root,
        the parent is set to None. If it is a leave the children list is empty.

        :param state: The state this node represents
        :param parent: A reference to the parent of this node
        """
        if children is None:
            children = []
        self.state: State = state
        self.__parent: Union[PrefixTree, None] = parent
        self.__children: List[PrefixTree] = children

    def set_state(self, state: State):
        """
        Method to set the state of the node.

        :param state: State to be set
        """
        self.state = state

    def get_children(self) -> List[PrefixTree]:
        """
        Method to get children of the node.

        :return: This nodes children
        """
        return self.__children

    def get_parent(self) -> PrefixTree:
        """
        Method to get the parent of the node.

        :return: This node's parent.
        """
        return self.__parent

    def add_child(self, child: PrefixTree):
        """
        Method to add a child to the node.

        :param child: Child to be added
        """
        self.__children.append(child)

    def copy(self):
        """
        Returns a shallow copy of this tree.
        :return: the copy tree
        """

        return PrefixTree(self.state, self.__parent, self.__children)

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()

    def depth(self) -> int:
        """
        Recursively calculates the depth of this tree.
        The runtime is O(n) where n is the number of nodes in this tree.
        :return: the depth as an int
        """

        return 1 + max([child.depth() for child in self.__children]) if self.__children else 1

    def merge(self, other: PrefixTree):
        """
        Merges a tree recursively into this instance from the root and returns the union tree.
        The tree's children are appended to this one's, the tree's parent is discarded.
        Requires the input tree to have the same root as this one.
        Assumes the tree is coherent: there are no duplicated children in any node.

        If the input tree is linear (every node has only 1 child) then the worst-case time complexity is O(n*m),
        where n is the maximum number of children of any node in this tree
        and m is the depth of the other tree.

        :param other: the tree to be merged into this one.
        :return: the resulting union tree.
        """

        assert other.state == self.state, "Trees do not have the same root!"

        result = PrefixTree(self.state, self.__parent)
        children = { hash(child.state) : child for child in self.__children }

        for child in other.get_children():
            child_hash = hash(child.state)
            if child_hash in children:
                children[child_hash] = children[child_hash].merge(child)
            else:
                children[child_hash] = child

        [result.add_child(child) for child in list(children.values())]

        return result
