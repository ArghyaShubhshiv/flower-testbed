import argparse
import os
from pathlib import Path

import flwr as fl
import numpy as np

from data_utils import load_data


class LinearRegressionModel:
    """Simple linear regression with explicit NumPy parameters.

    Keeping weights outside a sklearn estimator makes federated parameter
    exchange deterministic: Flower sends/receives exactly these arrays.
    """

    def __init__(self, n_features: int):
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = np.array([0.0], dtype=np.float64)

    def get_parameters(self):
        return [self.weights.copy(), self.bias.copy()]

    def set_parameters(self, parameters):
        weights, bias = parameters
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
        self.bias = np.asarray(bias, dtype=np.float64).reshape(1).copy()

    def predict(self, X):
        return X @ self.weights + self.bias[0]

    def fit(self, X, y, epochs=25, learning_rate=0.01):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        n = len(X)
        if n == 0:
            return

        for _ in range(epochs):
            pred = self.predict(X)
            error = pred - y
            grad_w = (X.T @ error) / n
            grad_b = np.mean(error)

            self.weights -= learning_rate * grad_w
            self.bias[0] -= learning_rate * grad_b

    def mse(self, X, y):
        error = self.predict(X) - np.asarray(y, dtype=np.float64).reshape(-1)
        return float(np.mean(error ** 2))


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_parameters()

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)

        epochs = int(config.get("local_epochs", 25))
        learning_rate = float(config.get("learning_rate", 0.01))

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        loss = self.model.mse(self.X_train, self.y_train)
        return (
            self.model.get_parameters(),
            len(self.X_train),
            {"train_mse": loss},
        )

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        loss = self.model.mse(self.X_test, self.y_test)
        return loss, len(self.X_test), {"mse": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.client_id)
    model = LinearRegressionModel(n_features=X_train.shape[1])

    server_address = os.environ.get("FL_SERVER_ADDRESS", "server:8080")
    ca_path = Path("certificates/ca.crt")
    root_certificates = ca_path.read_bytes() if ca_path.exists() else None

    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        ).to_client(),
        root_certificates=root_certificates,
        grpc_max_message_length=1024 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
