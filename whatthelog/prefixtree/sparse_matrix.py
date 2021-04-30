import bisect
from typing import Union, Tuple, Any, List


class SparseMatrix:
    """
    As we had no efficient way to store the edges, we implemented our own approach to achieve this.
    """

    separator = '.'

    def __init__(self):
        # The list in which the sparse matrix is stored
        self.list: List[str] = list()
        # For time efficiency purposes, we keep track of our own size
        self.size: int = 0

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set item in SparseMatrix using insertion sort.
        """
        tup = self.binary_search(key)
        if tup is None:
            bisect.insort_right(self.list, str(key[0]) + self.separator + str(key[1]) + self.separator + str(value))
            self.size += 1
        else:
            self.list[tup[0]] = str(key[0]) + self.separator + str(key[1]) + self.separator + str(value)

    def __getitem__(self, key: Tuple[int, int]) -> str:
        """
        Get item from SparseMatrix using binary search.
        """
        index: Union[Tuple[int, str], None] = self.binary_search(key)
        if index is None:
            raise KeyError
        return index[1]

    def __contains__(self, item: Tuple[int, int]) -> bool:
        """
        Checks if an edge exists.
        """
        index: int = self.bisearch(self.list, str(item[0]) + self.separator + str(item[1]))
        if index != self.size and self.get_value(index, True) == str(item[0]) + self.separator + str(item[1]):
            return True
        return False

    def __binary_search(self, search_list: List[str], item: Tuple[int, int]) -> Union[Tuple[int, str], None]:
        """
        Use binary search to search for a specific entry in the list of the SparseMatrix.
        """
        index: int = self.bisearch(search_list, str(item[0]) + self.separator + str(item[1]) + self.separator)
        if index != self.size:
            return index, self.get_value(index)
        else:
            return None

    def binary_search(self, item: Tuple[int, int]) -> Union[Tuple[int, str], None]:
        """
        Use binary search to search for a specific entry in the given list.
        """
        return self.__binary_search(self.list, item)

    def __binary_search_partial(self, search_list: List[str], item: int) -> Union[Tuple[int, str], None]:
        """
        Use binary search to search for a partial entry in the given list.
        """
        index: int = self.bisearch(search_list, str(item) + self.separator)
        if index != self.size:
            return index, self.get_value(index)
        else:
            return None

    def binary_search_partial(self, item: int) -> Union[Tuple[int, str], None]:
        """
        Use binary search to search for a partial entry in the list of the SparseMatrix.
        """
        return self.__binary_search_partial(self.list, item)

    def bisearch(self, arr: List[str], target: str) -> int:
        """
        Perform a binary search on the prefix (of unspecified length) of the elements
        """
        low: int = 0
        high: int = self.size - 1
        while high >= low:
            ix = (low + high) // 2
            if arr[ix].startswith(target):  # todo make this more efficient maybe?
                return ix
            elif target < arr[ix]:
                high = ix - 1
            else:
                low = ix + 1

        return self.size

    def get_value(self, index: int, key: bool = False) -> str:
        """
        Retrieves the value part of the SparseMatrix entry (or the key if key is True).
        """
        first: bool = False
        value: str = self.list[index]
        for j in range(len(value)):
            if first and value[j] == self.separator:
                if key:
                    return value[:j]
                else:
                    return value[j + 1:]
            if value[j] == self.separator:
                first = True
