import xgboost as xgb

from src.main import Main


class XgboostModel(Main):
    def __init__(self):
        self.model = xgb.XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return f"Accuracy: {accuracy:.4f}"
