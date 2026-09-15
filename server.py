from pathlib import Path

import flwr as fl
import numpy as np


NUM_CLIENTS = 2
NUM_FEATURES = 8  # California Housing has 8 input features
NUM_ROUNDS = 5


def main():
    cert_dir = Path("certificates")
    ca = cert_dir / "ca.crt"
    server_cert = cert_dir / "server.pem"
    server_key = cert_dir / "server.key"

    certificates = None
    if ca.exists() and server_cert.exists() and server_key.exists():
        certificates = (
            ca.read_bytes(),
            server_cert.read_bytes(),
            server_key.read_bytes(),
        )

    initial_parameters = fl.common.ndarrays_to_parameters(
        [
            np.zeros(NUM_FEATURES, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        ]
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        certificates=certificates,
        grpc_max_message_length=1024 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
