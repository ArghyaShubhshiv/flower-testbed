import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(client_id, num_clients=2):
    """Load California Housing, split into train/test, then partition among clients."""
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Train/test split (80/20) – same for all clients, but each gets a subset of training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split training data into `num_clients` non‑overlapping parts (homogeneous partition)
    # For simplicity, we slice sequentially – in practice you might shuffle.
    client_data = []
    chunk_size = len(X_train) // num_clients
    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_clients - 1 else len(X_train)
        X_client = X_train[start:end]
        y_client = y_train[start:end]
        # Normalize using scaler fitted on this client's data (or use global scaler)
        scaler = StandardScaler()
        X_client_scaled = scaler.fit_transform(X_client)
        client_data.append((X_client_scaled, y_client, X_test, y_test))
    return client_data[client_id - 1]  # client_id starts at 1