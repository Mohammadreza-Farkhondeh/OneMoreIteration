from abc import ABC, abstractmethod

import pandas as pd


class BaseDataPreprocessor(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs preprocessing steps on the input data.

        Args:
            data (pd.DataFrame): The data to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned preprocessing steps from fit to transform new data.

        Assumes the Preprocessor has already been fitted using the fit method.

        Args:
            data (pd.DataFrame): The data to be transformed.

        Returns:
            pd.DataFrame: The transformed data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def handle_missing_values(
        self, data: pd.DataFrame, method: str = "mean"
    ) -> pd.DataFrame:
        """
        Handles missing values in the data using the specified method.

        Args:
            data (pd.DataFrame): The data with missing values.
            method (str, optional): The method for handling missing values (e.g., "mean", "median"). Defaults to "mean".

        Returns:
            pd.DataFrame: The data with missing values handled.
        """

        pass

    @abstractmethod
    def normalize(self, data: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """
        Normalizes the data using the specified method.

        Args:
            data (pd.DataFrame): The data to be normalized.
            method (str, optional): The method for normalization (e.g., "standard", "min-max"). Defaults to "standard".

        Returns:
            pd.DataFrame: The normalized data.
        """

        pass

    @abstractmethod
    def encode_categorical_features(
        self, data: pd.DataFrame, method: str = "one-hot"
    ) -> pd.DataFrame:
        """
        Encodes categorical features in the data using the specified method.

        Args:
            data (pd.DataFrame): The data with categorical features.
            method (str, optional): The method for encoding categorical features (e.g., "one-hot", "label"). Defaults to "one-hot".

        Returns:
            pd.DataFrame: The data with encoded categorical features.
        """

        pass
