import bisect


def get_value(value: str):
    first = False
    i = 0
    while i < len(value):
        if first and value[i] == '.':
            return value[i+1:]
        if value[i] == '.':
            first = True
        i += 1


class SparseMatrix:
    """
    As we had no efficient way to store the edges. We implemented our own approach
    """
    def __init__(self):
        self.list = list()
        self.size = 0

    def __setitem__(self, key, value):
        """
        Set item in SparseMatrix using insertion sort.
        """
        tup = self.binary_search(key)
        if tup is None:
            bisect.insort(self.list, str(key[0]) + '.' + str(key[1]) + '.' + str(value))
            self.size += 1
        else:
            self.list[tup[0]] = str(key[0]) + '.' + str(key[1]) + '.' + str(value)

    def __getitem__(self, key):
        """
        Get item from SparseMatrix using binary search.
        """
        i = self.binary_search(key)
        if i is None:
            raise KeyError
        return i[1]

    def binary_search(self, item):
        i = bisect.bisect_right(self.list, str(item[0]) + '.' + str(item[1]) + '.')
        if i != self.size:
            return i, get_value(self.list[i])
        else:
            return None
