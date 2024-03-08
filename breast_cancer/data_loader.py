import pandas as pd

from src.data.loaders import BaseDataLoader


class BreastCancerDataLoader(BaseDataLoader):
    """
    Loads and preprocesses the breast cancer dataset from UCI.
    """

    def __init__(self, data_path: str):
        """
        Initializes the data loader with the data path.

        Args:
            data_path (str): The path to the breast cancer dataset file (CSV).
        """

        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the breast cancer data from the specified path and performs basic preprocessing.

        Returns:
            pd.DataFrame: The loaded and preprocessed data.
        """

        data = pd.read_csv(self.data_path)

        return data
