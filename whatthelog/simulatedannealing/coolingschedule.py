import abc
from numpy import log, e


class CoolingSchedule(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def initial_temperature(self, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def chain_length(self, *args, **kwargs) -> int:
        pass

    @abc.abstractmethod
    def update_temperature(self, *args, **kwargs) -> float:
        pass


class SimpleSchedule(CoolingSchedule):
    def __init__(self, initial_temperature: float, a: float, chain_length: int):
        super().__init__()
        self.initial_temperature = initial_temperature
        self.chain_length = chain_length
        self.temperature = initial_temperature
        self.a = a

    def initial_temperature(self) -> float:
        return self.initial_temperature

    def chain_length(self) -> int:
        return self.chain_length

    def update_temperature(self):
        self.temperature *= self.a
        return self.temperature


class BonomiLuttonSchedule(SimpleSchedule):
    def __init__(self):
        super().__init__(initial_temperature=1,
                         a=0.925,
                         chain_length=6)


class LundySchedule(CoolingSchedule):
    def __init__(self, alpha: float, neighborhood_size: float, sample_ratio: float = 1):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.sample_ratio = sample_ratio
        self.alpha = alpha
        self.temperature = 1

    def initial_temperature(self) -> float:
        return 1

    def chain_length(self) -> int:
        return max(2, int(self.neighborhood_size * self.sample_ratio))

    def update_temperature(self, deviation: float) -> float:
        self.temperature /= (1 + self.alpha * self.temperature)
        return self.temperature


class AartsSchedule(CoolingSchedule):
    def __init__(self,
                 neighborhood_size: int,
                 sample_ratio: float,
                 delta: float,
                 acceptance_ratio: float = 0.8,
                 avg_increase: float = 1):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.sample_ratio = sample_ratio
        self.delta = delta
        self.acceptance_ratio = acceptance_ratio
        self.avg_increase = avg_increase
        self.temperature = 100000  # self.avg_increase / log(e, 1 / self.acceptance_ratio)

    def initial_temperature(self) -> float:
        return 100000  # self.avg_increase / log(e, 1 / self.acceptance_ratio)

    def chain_length(self) -> int:
        return max(2, int(self.neighborhood_size * self.sample_ratio))

    def update_temperature(self, deviation: float) -> float:
        self.temperature = self.temperature / ((1 + self.temperature * log(self.delta + 1)) / (3 * deviation))
        return self.temperature
