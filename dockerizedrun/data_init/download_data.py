import os
import sys

sys.path.append("/app")
from common.data_utils import load_full_trainset, load_full_testset  # noqa: E402

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR", "/data")
    print(f"Downloading CIFAR-100 into {data_dir} ...")
    load_full_trainset(data_dir)
    load_full_testset(data_dir)
    print("CIFAR-100 ready.")
