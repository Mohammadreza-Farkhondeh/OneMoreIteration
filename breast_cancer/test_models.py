import importlib

from sklearn.model_selection import train_test_split

from .data_loader import BreastCancerDataLoader
from .data_preprocessor import BreastCancerDataPreprocessor

TARGET_COLUMN = "Class"

data_loader = BreastCancerDataLoader(data_path="../data/breast_cancer.csv")
data_preprocessor = BreastCancerDataPreprocessor()

data = data_loader.load_data()

preprocessed_data = data_preprocessor.fit(data).transform(data)
train_data, test_data = train_test_split(
    preprocessed_data, test_size=0.2, random_state=42
)

X = train_data.drop(TARGET_COLUMN, axis=1)
y = train_data[TARGET_COLUMN]

X_test = test_data.drop(TARGET_COLUMN, axis=1)
y_test = test_data[TARGET_COLUMN]

models = {
    "PytorchModel": "model_pytorch_classification",
    "SklearnModel": "model_sklearn_classification",
    "TensorflowModel": "model_tensorflow_classification",
    "LogisticRegressionModel": "model_logistic_regression_classification",
    "SVModel": "model_svm_classification",
    "DesicionTreeModel": "model_decision_tree_classification",
    "RandomForestModel": "model_random_forest_classification",
    "KNeighborsModel": "model_kneighbors_classification",
    "NaiveBayesModel": "model_naive_bayes_classification",
    "XgboostModel": "model_xgboost_classification",
    "LightgbmModel": "model_lightgbm_classification",
    "CatboostModel": "model_catboost_classification",
    "H2oModel": "model_h2o_classification",
    "PycaretModel": "model_pycaret_classification",
}

for model, module_name in models.items():
    try:
        model_module = importlib.import_module(module_name)
        classifier = getattr(model_module, model)()
        classifier.fit(X, y)
        evaluation_metrics = classifier.evaluate(X_test, y_test)
        print(f"Evaluation Metrics: {evaluation_metrics}")
    except ImportError as ie:
        print(f"Import error: {ie}")
    except Exception as e:
        print(f"Exception: {e}")
