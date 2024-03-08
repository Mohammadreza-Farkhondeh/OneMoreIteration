import main

from src.main import Main

from .data_loader import BreastCancerDataLoader
from .model_pytorch_regression import PytorchBreastCancerModel
from .model_sklearn_regression import SklearnBreastCancerModel
from .model_tensorflow_regression import TensorflowBreastCancerModel

if __name__ == "__main__":
    data_loader = BreastCancerDataLoader(data_path="../data/breast_cancer.csv")
    data_preprocessor = BreastCancerDataPreprocessor()
    model_sklearn = SklearnBreastCancerModel()
    model_pytorch = PytorchBreastCancerModel()
    model_tensorflow = TensorflowBreastCancerModel()

    main = Main(
        data_loader=data_loader, preprocessor=data_preprocessor, model=model_sklearn
    )
    main.run()
