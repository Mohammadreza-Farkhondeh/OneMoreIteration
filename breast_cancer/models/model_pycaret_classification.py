from pycaret.classification import *


class PycaretModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = setup(data=X, target=y, session_id=42)
        self.best_model = compare_models()
        self.best_model.fit(X, y)

    def evaluate(self, X_test, y_test):
        df = evaluate_model(self.best_model, data=X_test)
        accuracy = df["Accuracy"]
        return f"Accuracy: {accuracy:.4f}"

    def __del__(self):
        finalize_model()
