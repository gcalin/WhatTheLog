from whatthelog.prefixtree.state import State


def test_id():
    x = State(["root"])
    y = State(["root"])

    assert id(x) != id(y)
    assert id(x) == id(x)

    oldx = id(x)
    x.is_terminal = True
    assert id(x) == oldx

    x = State(['other state'])
    assert id(x) != oldx


def test_is_equivalent():
    x = State(["root"])
    y = State(["root"])
    z = State(["root", "other"])

    assert x.is_equivalent(y)
    assert not x.is_equivalent(z)
