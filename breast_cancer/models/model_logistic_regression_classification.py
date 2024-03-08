from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self):
        self.classifier = LogisticRegression()

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def evaluate(self, X_test, y_test):
        y_pred = self.classifier.predict(X_test)
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_test, y_pred)
        return f"Accuracy: {accuracy:.4f}"
