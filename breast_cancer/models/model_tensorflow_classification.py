import tensorflow as tf

<<<<<<< HEAD
from src.main import Main
=======
from src.models import BaseModel
>>>>>>> a979fe0884f26982df2a6c8345191001a9bdd8b3


class Model(tf.Module):
    pass


<<<<<<< HEAD
class TensorflowModel(Main):
=======
class TensorflowModel(BaseModel):
>>>>>>> a979fe0884f26982df2a6c8345191001a9bdd8b3
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
