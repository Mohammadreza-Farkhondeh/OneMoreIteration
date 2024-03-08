from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series or np.ndarray) -> None:
        """
        Trains the model on the provided data.

        Args:
            X (pd.DataFrame): The features data.
            y (pd.Series or np.ndarray): The target labels.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series or np.ndarray:
        """
        Makes predictions on new data using the trained model.

        Args:
            X (pd.DataFrame): The data for predictions.

        Returns:
            pd.Series or np.ndarray: The predicted labels or values.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series or np.ndarray) -> dict:
        """
        Evaluates the model's performance on the provided data.

        Args:
            X (pd.DataFrame): The features data for evaluation.
            y (pd.Series or np.ndarray): The true target labels.

        Returns:
            dict: A dictionary containing evaluation metrics (e.g., accuracy, precision, recall, F1-score).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError("Subclasses must implement this method")

    def save(self, path: str) -> None:
        """
        Saves the trained model to disk.

        Args:
            path (str): The path to save the model.
        """

        pass

    def load(self, path: str) -> None:
        """
        Loads a pre-trained model from disk.

        Args:
            path (str): The path to load the model from.
        """

        pass
