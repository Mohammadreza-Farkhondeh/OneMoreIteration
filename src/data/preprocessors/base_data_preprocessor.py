from abc import ABC, abstractmethod

from src.utils.logger import Logger


class BaseDataPreprocessor(ABC):
    def __init__(self, *args, **kwargs):
        self.logger = Logger().get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} data preprocessor")

    @abstractmethod
    def preprocess_data(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def clean_data(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def transform_data(self, data, *args, **kwargs):
        pass
