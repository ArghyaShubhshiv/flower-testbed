import flwr as fl

def main():
    # Start Flower server with FedAvg strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
        )
    )

if __name__ == "__main__":
    main()