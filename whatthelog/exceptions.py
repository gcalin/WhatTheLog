from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnidentifiedLogException(Exception):
    message: str = field(default="Log message not found in syntax tree!")

@dataclass(frozen=True)
class InvalidTreeException(Exception):
    message: str = field(default="Tree is invalid")

@dataclass(frozen=True)
class StateAlreadyExistsException(Exception):
    message: str = field(default="Attempted to add duplicate state to tree!")