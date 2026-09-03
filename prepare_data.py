import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

NUM_CLIENTS = 2
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

chunk_size = len(X_train) // NUM_CLIENTS
for i in range(NUM_CLIENTS):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != NUM_CLIENTS - 1 else len(X_train)
    scaler = StandardScaler()
    X_client = scaler.fit_transform(X_train[start:end])
    y_client = y_train[start:end]

    out_dir = f"data/client{i+1}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X_train.npy", X_client)
    np.save(f"{out_dir}/y_train.npy", y_client)
    np.save(f"{out_dir}/X_test.npy", X_test)
    np.save(f"{out_dir}/y_test.npy", y_test)
