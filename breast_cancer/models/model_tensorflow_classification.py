import tensorflow as tf

from src.models import BaseModel


class Model(tf.Module):
    pass


class TensorflowModel(BaseModel):
    def __init__(self):
        self.model = Model()
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def fit(self, X, y):
        self.model.fit(X, y, epochs=100, batch_size=32)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return f"Accuracy: {accuracy:.4f}"
