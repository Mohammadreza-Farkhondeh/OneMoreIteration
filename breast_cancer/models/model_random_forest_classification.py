from sklearn.ensemble import RandomForestClassifier

from src.models import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self):
        self.classifier = RandomForestClassifier()

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def evaluate(self, X_test, y_test):
        y_pred = self.classifier.predict(X_test)
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_test, y_pred)
        return f"Accuracy: {accuracy:.4f}"
