from abc import ABC, abstractmethod

from src.utils.logger import Logger


class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        self.logger = Logger().get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} model")

    @abstractmethod
    def train(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data, *args, **kwargs):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def save(self, *args, **kwargs):
        pass
