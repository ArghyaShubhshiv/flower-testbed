from pathlib import Path

import numpy as np


DATA_ROOT = Path("data")


def load_data(client_id: int):
    """Load the shard mounted for one client.

    Docker mounts data/clientN to /app/data, so the files below are always
    available as /app/data/X_train.npy, etc.
    """
    if client_id < 1:
        raise ValueError("client_id must be >= 1")

    paths = {
        "X_train": DATA_ROOT / "X_train.npy",
        "y_train": DATA_ROOT / "y_train.npy",
        "X_test": DATA_ROOT / "X_test.npy",
        "y_test": DATA_ROOT / "y_test.npy",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files: " + ", ".join(missing) +
            ". Run prepare_data.py before starting the clients."
        )

    X_train = np.load(paths["X_train"])
    y_train = np.load(paths["y_train"])
    X_test = np.load(paths["X_test"])
    y_test = np.load(paths["y_test"])

    return X_train, y_train, X_test, y_test
