import main

from src.main import Main

from .data_loader import BreastCancerDataLoader
from .data_preprocessor import BreastCancerDataPreprocessor
from .model_sklearn_classification import SklearnBreastCancerModel

if __name__ == "__main__":
    data_loader = BreastCancerDataLoader(data_path="../data/breast_cancer.csv")
    data_preprocessor = BreastCancerDataPreprocessor()
    model = SklearnBreastCancerModel()

    main = Main(
        data_loader=data_loader,
        preprocessor=data_preprocessor,
        model=model,
        target_column="Class",
    )
    main.run()
