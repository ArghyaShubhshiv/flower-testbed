import flwr as fl
from pathlib import Path

def main():
    certificates = (
        Path("certificates/ca.crt").read_bytes(),
        Path("certificates/server.pem").read_bytes(),
        Path("certificates/server.key").read_bytes(),
    )
    #fedavg
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
        ),
        certificates=certificates,
    )

if __name__ == "__main__":
    main()
