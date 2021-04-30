import bisect
from typing import Union, Tuple, Any, List


class SparseMatrix:
    """
    As we had no efficient way to store the edges, we implemented our own approach to achieve this.
    """

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
            bisect.insort(self.list, str(key[0]) + '.' + str(key[1]) + '.' + str(value))
            self.size += 1
        else:
            self.list[tup[0]] = str(key[0]) + '.' + str(key[1]) + '.' + str(value)

    def __getitem__(self, key: Tuple[int, int]) -> str:
        """
        Get item from SparseMatrix using binary search.
        """
        i: Union[Tuple[int, str], None] = self.binary_search(key)
        if i is None:
            raise KeyError
        return i[1]

    def binary_search(self, item: Tuple[int, int]) -> Union[Tuple[int, str], None]:
        """
        Use binary search to search for a specific entry of the sparse matrix in the list.
        """
        i: int = bisect.bisect_right(self.list, str(item[0]) + '.' + str(item[1]) + '.')
        if i != self.size:
            return i, self.get_value(i)
        else:
            return None

    def get_value(self, i: int) -> str:
        """
        Retrieves the value part of the SparseMatrix entry.
        """
        first: bool = False
        value: str = self.list[i]
        for j in range(len(value)):
            if first and value[j] == '.':
                return value[j + 1:]
            if value[j] == '.':
                first = True
