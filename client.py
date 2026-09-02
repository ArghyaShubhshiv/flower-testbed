import argparse
import warnings
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import flwr as fl
from data_utils import load_data

warnings.filterwarnings("ignore")

class SklearnClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # Store feature count for zero-initialization
        self.n_features = X_train.shape[1]

    def get_parameters(self, config):
        # If the model is not yet fitted, return zeros
        if not hasattr(self.model, 'coef_'):
            # Return zero coefficients and intercept
            coef = np.zeros((self.n_features, 1))
            intercept = np.zeros((1, 1))
        else:
            coef = self.model.coef_.reshape(-1, 1)
            intercept = np.array([[self.model.intercept_]])
        return [coef, intercept]

    def set_parameters(self, parameters):
        coef = parameters[0].reshape(-1)
        intercept = parameters[1].item()
        self.model.coef_ = coef
        self.model.intercept_ = intercept

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return float(mse), len(self.X_test), {"mse": mse}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (1,2,...)")
    args = parser.parse_args()

    # Load data for this client
    X_train, y_train, X_test, y_test = load_data(args.client_id, num_clients=2)

    model = LinearRegression()

    # Determine server address: use environment variable if set, otherwise default
    server_address = os.environ.get("FL_SERVER_ADDRESS", "server:8080")

    # Start Flower client (using the deprecated API for now – it works)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=SklearnClient(model, X_train, y_train, X_test, y_test),
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    main()