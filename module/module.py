from abc import ABC, abstractmethod

class Module(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dY):
        pass