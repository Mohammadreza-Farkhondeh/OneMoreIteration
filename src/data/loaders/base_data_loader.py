from abc import ABC, abstractmethod

from src.utils.logger import Logger


class BaseDataLoader(ABC):
    def __init__(self, *args, **kwargs):
        self.logger = Logger().get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} data loader")

    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def read_data(self, *args, **kwargs):
        pass
