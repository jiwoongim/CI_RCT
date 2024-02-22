from abc import ABC, abstractmethod

import numpy as np


RNG = np.random.RandomState()
ITEM = 0


class BanditArmABC(ABC):
    def __init__(self, num_action: int):
        super().__init__(num_action)

    @abstractmethod
    def sample_covariate(self):
        ...

    @abstractmethod
    def sample_action(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def sample_action_given_covariate(self, x: float) -> tuple[int, float]:
        ...

    @abstractmethod
    def sample_outcome(self, x: float, a: float) -> float:
        ...
