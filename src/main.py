from typing import List, Union

from sklearn.model_selection import train_test_split

from src.data.loaders import BaseDataLoader
from src.data.preprocessors import BaseDataPreprocessor
from src.models import BaseModel


class Main:
    def __init__(
        self,
        data_loader: BaseDataLoader,
        preprocessor: BaseDataPreprocessor,
        model: BaseModel,
        target_column: Union[List, str],
    ):
        """
        Initializes the ML project with data loader, preprocessor, and model objects.

        Args:
            data_loader (DataLoaderABC): The data loader object for loading data.
            preprocessor (PreprocessorABC): The preprocessor object for preprocessing data.
            model (ModelABC): The model object for training and prediction.
        """

        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = model
        self.target_column = target_column

    def run(self, data_path: str):
        """
        Runs the entire machine learning pipeline.

        Args:
            data_path (str): The path to the data source.

        Raises:
            ValueError: If any of the provided objects are not instances of their respective base classes.
        """

        if not isinstance(self.data_loader, BaseDataLoader):
            raise ValueError("data_loader must be an instance of BaseDataLoader")
        if not isinstance(self.preprocessor, BaseDataPreprocessor):
            raise ValueError("preprocessor must be an instance of BaseDataPreprocessor")
        if not isinstance(self.model, BaseModel):
            raise ValueError("model must be an instance of BaseModel")

        data = self.data_loader.load_data(data_path)

        preprocessed_data = self.preprocessor.fit(data).transform(data)
        train_data, test_data = train_test_split(
            preprocessed_data, test_size=0.2, random_state=42
        )

        X = train_data.drop(self.target_column, axis=1)
        y = train_data[self.target_column]

        self.model.fit(X, y)

        X_test = test_data.drop(self.target_column, axis=1)
        y_test = test_data[self.target_column]

        evaluation_metrics = self.model.evaluate(X_test, y_test)
        print(f"Evaluation Metrics: {evaluation_metrics}")

        return self.model
