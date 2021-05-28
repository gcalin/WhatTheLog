import abc
from numpy import log, e


class CoolingSchedule(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def initial_temperature(self, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def chain_length(self, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def update_temperature(self, *args, **kwargs):
        pass


class SimpleSchedule(CoolingSchedule):
    def __init__(self, initial_temperature: float, a: float, chain_length: int):
        super().__init__()
        self.initial_temperature = initial_temperature
        self.chain_length = chain_length
        self.temperature = initial_temperature
        self.a = a

    def initial_temperature(self, *args, **kwargs) -> float:
        return self.initial_temperature

    def chain_length(self, *args, **kwargs) -> int:
        return self.chain_length

    def update_temperature(self, *args, **kwargs):
        self.temperature *= self.a


class BonomiLuttonSchedule(SimpleSchedule):
    def __init__(self):
        super().__init__(initial_temperature=1,
                         a=0.925,
                         chain_length=6)


class LundySchedule(CoolingSchedule):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.temperature = 0.5

    def initial_temperature(self) -> float:
        return 0.5

    def chain_length(self) -> int:
        return 6

    def update_temperature(self, deviation: float):
        self.temperature /= (1 + self.alpha * self.temperature)


class AartsSchedule(CoolingSchedule):
    def __init__(self, avg_increase: float,
                 neighborhood_size: int,
                 delta: float,
                 acceptance_ratio: float = 0.8):
        super().__init__()
        self.avg_increase = avg_increase
        self.acceptance_ratio = acceptance_ratio
        self.neighborhood_size = neighborhood_size
        self.delta = delta

    def initial_temperature(self) -> float:
        return self.avg_increase / log(e, 1 / self.acceptance_ratio)

    def chain_length(self) -> int:
        return self.neighborhood_size

    def update_temperature(self, deviation: float):
        self.temperature = self.temperature / ((1 + self.temperature * log(e, self.delta + 1)) / (3 * deviation))
