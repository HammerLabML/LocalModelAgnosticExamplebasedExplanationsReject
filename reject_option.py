from abc import ABC, abstractmethod


class RejectOption(ABC):
    def __init__(self, threshold, **kwds):
        self.threshold = threshold

        super().__init__(**kwds)

    @abstractmethod
    def criterion(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.reject(x)

    def reject(self, x):
        return self.criterion(x) < self.threshold
