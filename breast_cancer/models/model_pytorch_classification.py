import torch
from torch import nn


class Model(nn.Module):
    pass


class PytorchModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = Model().to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters())

    def fit(self, X, y):
        X_tensor = torch.tensor(X).to(self.device)
        y_tensor = torch.tensor(y).to(self.device).float()
        for epoch in range(100):
            y_pred = self.classifier(X_tensor)
            loss = self.loss_fn(y_pred, y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, X_test, y_test):
        X_test_tensor = torch.tensor(X_test).to(self.device)
        y_test_tensor = torch.tensor(y_test).to(self.device).float()
        with torch.no_grad():
            y_pred = self.classifier(X_test_tensor)
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_test_tensor.cpu(), y_pred.cpu().detach().numpy())
        return f"AUC-ROC: {auc:.4f}"  # Assuming your model outputs probabilities
