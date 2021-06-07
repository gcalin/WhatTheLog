import abc
import random
from typing import Tuple, List

from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.state import State


class Selection(abc.ABC):
    def __init__(self, model: Graph):
        self.model = model

    @abc.abstractmethod
    def select(self, *args, **kwargs) -> Tuple[State, State]:
        pass

    def update(self, model: Graph):
        self.model = model


class RandomSelection(Selection):
    def __init__(self, model: Graph):
        super().__init__(model)

    def select(self) -> Tuple[State, State]:
        state = self.model.get_random_state()

        while len(self.model.get_outgoing_states_not_self(state)) == 0:
            state = self.model.get_random_state()

        neighbour = self.model.get_random_child(state)
        return state, neighbour


class RouletteWheelSelection(Selection):
    def __init__(self, model: Graph, k: int):
        super().__init__(model)
        self.k = k

    def select(self) -> Tuple[State, State]:
        # The array of randomly considered choices
        choices: List[Tuple[State, State, float]] = []

        # To total sum of the scores of the choices
        total: float = 0

        for _ in range(self.k):
            # Select at random a state and its neighbour
            state = self.model.get_random_state()
            neighbour = self.model.get_random_child(state)

            # Calculate the score between 0 and 1 (higher is better) and store in array
            score: float = 1 / len(self.model.get_outgoing_states(state))
            choices.append((state, neighbour, score))

            # Count the total score
            total += score

        population: List[Tuple[State, State]] = []
        weights: List[float] = []

        for count, tup in enumerate(choices):
            # Separate the population from the score
            population.append((tup[0], tup[1]))

            # Normalize the score into weights such that the sum is equal to 1
            weights.append(tup[2] / total)

        # Return a randomly chosen state and its child based on the weights
        return random.choices(
            population=population,
            weights=weights,
            k=1)[0]


class TournamentSelection(Selection):
    def __init__(self, model: Graph, k: int, p: float):
        super().__init__(model)
        self.k = k
        self.p = p

    def select(self) -> Tuple[State, State]:
        # The array of randomly considered choices
        choices: List[Tuple[State, State, float]] = []

        for _ in range(self.k):
            # Select at random a state and its neighbour
            state, neighbour = self.model.get_non_terminal_state_and_child()

            # Calculate the score between 0 and 1 (higher is better) and store in array
            score: float = 1 / len(self.model.get_outgoing_states(state))
            choices.append((state, neighbour, score))

        # Sort the array descending on score
        sorted(choices, key=lambda tup: tup[2], reverse=True)

        # Initialize probability
        probability: float = self.p

        for tup in choices:
            # Select with probability p*(1-p)^(i-1)
            if random.random() < probability:
                return tup[0], tup[1]
            # Update probability
            probability *= 1-self.p

        # If no choice made, return the best candidate
        return choices[0][0], choices[0][1]
