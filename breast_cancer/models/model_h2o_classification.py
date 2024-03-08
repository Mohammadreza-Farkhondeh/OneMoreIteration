import h2o


class H2oModel:
    def __init__(self):
        h2o.init()

    def fit(self, X, y):
        self.model = h2o.random_forest(training_frame=h2o.H2OFrame(X), y=y)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(h2o.H2OFrame(X_test))

        accuracy_value = h2o.metrics.accuracy(y_test, y_pred)
        return f"Accuracy: {accuracy_value:.4f}"

    def __del__(self):
        h2o.shutdown()
