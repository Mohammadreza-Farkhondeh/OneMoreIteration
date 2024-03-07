from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import pandas as pd


class BaseDataLoader(ABC):
    @abstractmethod
    def load_data(self, file: Union[str, List, Path], *args, **kwargs) -> pd.DataFrame:
        """
        Loads data from the specified path and returns a Pandas DataFrame.

        Args:
            data_path (str): The path to the data source (file, URL, etc.).

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass
