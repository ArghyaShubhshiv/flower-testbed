import os

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NUM_CLIENTS = 2
RANDOM_STATE = 42


def main():
    data = fetch_california_housing()
    X, y = data.data.astype(np.float64), data.target.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Fit ONE scaler using the global training data so every client and the
    # shared test set use the same feature transformation.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    splits = np.array_split(np.arange(len(X_train)), NUM_CLIENTS)

    for client_id, indices in enumerate(splits, start=1):
        out_dir = f"data/client{client_id}"
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"{out_dir}/X_train.npy", X_train[indices])
        np.save(f"{out_dir}/y_train.npy", y_train[indices])
        np.save(f"{out_dir}/X_test.npy", X_test)
        np.save(f"{out_dir}/y_test.npy", y_test)

        print(f"client{client_id}: {len(indices)} training samples")


if __name__ == "__main__":
    main()
