import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.preprocessors import BaseDataPreprocessor


class BresstCancerDataPreprocessor(BaseDataPreprocessor):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.handle_missing_values(data)
        data = self.encode_categorical_features(data)
        data = self.normalize(data)
        return data

    def handle_missing_values(
        self, data: pd.DataFrame, method: str = "mean"
    ) -> pd.DataFrame:
        data.replace("?", pd.NA, inplace=True)
        data_cleaned = data.dropna()
        return data_cleaned

    def normalize(self, data: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        numeric_columns = ["tumor-size", "inv-nodes", "deg-malig"]

        if method == "standard":
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        return data

    def encode_categorical_features(
        self, data: pd.DataFrame, method: str = "one-hot"
    ) -> pd.DataFrame:
        if method == "one-hot":
            data_encoded = pd.get_dummies(
                data, columns=data.select_dtypes(include=["object"]).columns
            )
        else:
            raise ValueError(f"Unsupported encoding method: {method}")

        return data_encoded

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data
