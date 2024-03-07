from src.data.loaders import BaseDataLoader
from src.data.preprocessors import BaseDataPreprocessor
from src.models import BaseModel


class Main:
    def __init__(
        self,
        data_loader: BaseDataLoader,
        preprocessor: BaseDataPreprocessor,
        model: BaseModel,
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

        # Load data
        data = self.data_loader.load_data(data_path)

        # Preprocess data (assuming fit and transform are implemented)
        preprocessed_data = self.preprocessor.fit(data).transform(data)

        # Separate features and target variables
        X = preprocessed_data.drop(
            "target_column", axis=1
        )  # Replace 'target_column' with your actual target column name
        y = preprocessed_data["target_column"]

        # Train the model
        self.model.fit(X, y)

        # Optionally evaluate the model
        evaluation_metrics = self.model.evaluate(X, y)
        print(f"Evaluation Metrics: {evaluation_metrics}")

        # Optionally make predictions on new data (replace with your prediction logic)
        # new_data = ...  # Load or prepare new data
        # predictions = self.model.predict(new_data)
        # print(f"Predictions: {predictions}")
