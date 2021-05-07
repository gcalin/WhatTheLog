from whatthelog.prefixtree.state import State


def test_hash():
    x = State(["root"])
    y = State(["root"])

    assert hash(x) != hash(y)


def test_is_equivalent():
    x = State(["root"])
    y = State(["root"])
    z = State(["root", "other"])

    assert x.is_equivalent(y)
    assert x.is_equivalent(z) is False