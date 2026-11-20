"""CIFAR-10: download once, store as uint8 .npy, fixed IID partition.

    python -m hwsig.data --root data/cifar10 --n-clients 6        # real data
    python -m hwsig.data --root data/fake10 --n-clients 6 --fake   # smoke test

The partition is fixed (seed 0) across *all* runs, so honest and attack runs
see byte-identical shards; only the training seed changes between pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _load_batches(tar_path: Path):
    xs, ys, xt, yt = [], [], None, None
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if not (name.startswith("data_batch") or name == "test_batch"):
                continue
            d = pickle.load(tf.extractfile(m), encoding="bytes")
            x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
            y = np.array(d[b"labels"], dtype=np.int64)
            if name == "test_batch":
                xt, yt = x, y
            else:
                xs.append(x)
                ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), xt, yt


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(tar: Path, tries: int = 3) -> Path:
    """Download to .part, verify MD5, then rename -- a partial file is never reused."""
    if tar.exists() and _md5(tar) == MD5:
        return tar
    if tar.exists():
        print(f"{tar} is corrupt or incomplete; re-downloading")
        tar.unlink()
    part = tar.with_suffix(".part")
    for i in range(tries):
        print(f"downloading {URL} (attempt {i + 1}/{tries}) ...", flush=True)
        try:
            urllib.request.urlretrieve(URL, part)
            if _md5(part) == MD5:
                part.rename(tar)
                return tar
            print("checksum mismatch")
        except OSError as e:
            print(f"download failed: {e}")
        if part.exists():
            part.unlink()
    raise SystemExit(f"could not download CIFAR-10; fetch {URL} manually into {tar}")


def prepare(root: str, n_clients: int, fake: bool = False, seed: int = 0,
            fake_train: int = 6000, fake_test: int = 1000) -> None:
    rootp = Path(root)
    rootp.mkdir(parents=True, exist_ok=True)
    if not (rootp / "X_train.npy").exists():
        if fake:
            rng = np.random.default_rng(1234)
            # weakly learnable: class-dependent mean colour + noise
            def synth(n):
                y = rng.integers(0, 10, n)
                base = (y[:, None, None, None] * 23 % 255).astype(np.int16)
                x = np.clip(base + rng.normal(0, 60, (n, 32, 32, 3)), 0, 255).astype(np.uint8)
                return x, y.astype(np.int64)
            Xtr, ytr = synth(fake_train)
            Xte, yte = synth(fake_test)
        else:
            tar = _download(rootp / "cifar-10-python.tar.gz")
            Xtr, ytr, Xte, yte = _load_batches(tar)
        np.save(rootp / "X_train.npy", Xtr)
        np.save(rootp / "y_train.npy", ytr)
        np.save(rootp / "X_test.npy", Xte)
        np.save(rootp / "y_test.npy", yte)
    ytr = np.load(rootp / "y_train.npy")
    part = rootp / f"partition_n{n_clients}_seed{seed}.npz"
    if not part.exists():
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ytr))
        shards = np.array_split(idx, n_clients)
        np.savez(part, **{f"c{i}": s for i, s in enumerate(shards)})
    print(f"ready: {rootp}  ({len(ytr)} train, partition {part.name})")


def to_tensor_ready(x_uint8: np.ndarray) -> np.ndarray:
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def load_client(root: str, n_clients: int, cid: int, max_samples: int = 0, seed: int = 0):
    rootp = Path(root)
    X = np.load(rootp / "X_train.npy", mmap_mode="r")
    y = np.load(rootp / "y_train.npy")
    idx = np.load(rootp / f"partition_n{n_clients}_seed{seed}.npz")[f"c{cid}"]
    if max_samples:
        idx = idx[:max_samples]
    return to_tensor_ready(np.asarray(X[idx])), y[idx]


def load_test(root: str, max_samples: int = 0):
    rootp = Path(root)
    X = np.load(rootp / "X_test.npy")
    y = np.load(rootp / "y_test.npy")
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]
    return to_tensor_ready(X), y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cifar10")
    ap.add_argument("--n-clients", type=int, default=6)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    prepare(a.root, a.n_clients, fake=a.fake)
