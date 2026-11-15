"""
CIFAR-100 loading + Dirichlet-based non-IID partitioning across clients.

Partitioning strategy (Hsu et al., 2019 style) -- UNCHANGED from the Docker version:
  For each class c, draw a proportion vector p ~ Dirichlet(alpha, ..., alpha)
  over the N clients, and split that class's examples across clients
  according to p. Any client below `min_samples_per_client` is topped up
  from the clients with the largest surplus.

  alpha -> 0   : each client sees (close to) a single class -> highly non-IID
  alpha -> inf : partition approaches uniform/IID across clients

What changed: the partition is now computed ONCE by the launcher and each
client's shard is written to its own folder:

    client_data/
      partition_config.json          # config + per-client sizes (cache key)
      client_0/train.npz             # x: uint8 [N,32,32,3], y: int64 [N], indices: global train ids
      client_0/meta.json             # size, #classes present, per-class counts
      client_1/...

A client thread only ever reads its own folder. The official CIFAR-100 test
split (10k images) is NEVER partitioned to clients; it is reserved for the
server's centralized evaluation.

Batches are served from in-memory tensors, with RandomCrop(32, pad=4) +
horizontal flip + normalization done on whole batches (same augmentation as
the old torchvision pipeline, but without per-sample PIL work or DataLoader
worker processes, which is what you want when many clients share one machine).

Only numpy is needed to partition; torch is imported lazily by the loaders.
"""
import json
import math
import os
import pickle
import re
import shutil

import numpy as np

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
PARTITION_FILE = "partition_config.json"


# --------------------------------------------------------------------------- raw data
def _cifar_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "cifar-100-python")


def _ensure_downloaded(data_dir: str):
    needed = ("train", "test", "meta")
    if all(os.path.exists(os.path.join(_cifar_dir(data_dir), f)) for f in needed):
        return
    print(f"Downloading CIFAR-100 into {data_dir} ...")
    from torchvision import datasets  # lazy: only needed for the one-time download
    datasets.CIFAR100(root=data_dir, train=True, download=True)
    datasets.CIFAR100(root=data_dir, train=False, download=True)


def load_raw_cifar100(data_dir: str, train: bool):
    """Return (images uint8 [N,32,32,3], labels int64 [N]) from the standard CIFAR-100 pickle files."""
    _ensure_downloaded(data_dir)
    path = os.path.join(_cifar_dir(data_dir), "train" if train else "test")
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    x = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.asarray(entry["fine_labels"], dtype=np.int64)
    return np.ascontiguousarray(x), y


# --------------------------------------------------------------------------- partitioning (verbatim)
def _redistribute_for_min_samples(client_indices, min_samples: int, rng: np.random.Generator):
    """
    Top up any client below `min_samples` by moving samples from clients
    that have surplus above `min_samples`. Guarantees every client ends up
    with >= min_samples, as long as total_samples >= min_samples * num_clients.
    This is what actually prevents the "some clients get ~0 data" failure
    mode of raw Dirichlet sampling at low alpha.
    """
    num_clients = len(client_indices)
    total = sum(len(x) for x in client_indices)
    if min_samples * num_clients > total:
        raise ValueError(
            f"min_samples_per_client={min_samples} * num_clients={num_clients} "
            f"= {min_samples * num_clients} exceeds the {total} available training "
            f"examples. Lower --min-samples-per-client or reduce --num-clients."
        )

    client_indices = [list(x) for x in client_indices]

    for cid in range(num_clients):
        deficit = min_samples - len(client_indices[cid])
        while deficit > 0:
            # Pick whichever client currently has the largest surplus to donate from.
            donor = max(
                (c for c in range(num_clients) if c != cid),
                key=lambda c: len(client_indices[c]) - min_samples,
            )
            surplus = len(client_indices[donor]) - min_samples
            if surplus <= 0:
                # Shouldn't happen given the total-capacity check above, but
                # guard against infinite loop just in case of rounding edge cases.
                break
            take = min(deficit, surplus)
            rng.shuffle(client_indices[donor])
            moved, client_indices[donor] = client_indices[donor][:take], client_indices[donor][take:]
            client_indices[cid].extend(moved)
            deficit -= take

    return client_indices


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int = 42,
                         min_samples_per_client: int = 20):
    """
    Return a list of `num_clients` index lists partitioning `labels`.

    Raw per-class Dirichlet sampling can leave some clients with very few or
    even zero examples (more likely at low alpha / many clients), which
    breaks training for that client. After the Dirichlet split, any client
    below `min_samples_per_client` is topped up by transferring examples
    from clients with a surplus (see `_redistribute_for_min_samples`), so
    every client is guaranteed at least `min_samples_per_client` examples
    while the overall distribution still reflects the requested alpha.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    if min_samples_per_client > 0:
        client_indices = _redistribute_for_min_samples(client_indices, min_samples_per_client, rng)

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    return client_indices


# --------------------------------------------------------------------------- per-client folders
def client_dir_name(client_id: int) -> str:
    return f"client_{client_id}"


def _partition_is_current(clients_dir: str, config: dict) -> bool:
    cfg_path = os.path.join(clients_dir, PARTITION_FILE)
    if not os.path.exists(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return False
    if saved.get("config") != config:
        return False
    return all(
        os.path.exists(os.path.join(clients_dir, client_dir_name(i), "train.npz"))
        for i in range(config["num_clients"])
    )


def partition_to_folders(data_dir: str, clients_dir: str, num_clients: int, alpha: float,
                         seed: int = 42, min_samples_per_client: int = 20, force: bool = False):
    """
    Partition the CIFAR-100 TRAIN split and write one folder per client.
    Skips the work if `clients_dir` already holds the same partition
    (same num_clients / alpha / seed / min_samples); pass force=True to redo it.
    Returns the list of per-client sample counts.
    """
    config = {"dataset": "cifar100", "num_clients": num_clients, "alpha": alpha,
              "seed": seed, "min_samples_per_client": min_samples_per_client}
    if not force and _partition_is_current(clients_dir, config):
        with open(os.path.join(clients_dir, PARTITION_FILE)) as f:
            sizes = json.load(f)["client_sizes"]
        print(f"[data] Reusing existing partition in {clients_dir}/ ({num_clients} clients, alpha={alpha})")
        return sizes

    x, y = load_raw_cifar100(data_dir, train=True)
    partitions = dirichlet_partition(y, num_clients, alpha, seed, min_samples_per_client)

    os.makedirs(clients_dir, exist_ok=True)
    cfg_path = os.path.join(clients_dir, PARTITION_FILE)
    if os.path.exists(cfg_path):
        os.remove(cfg_path)  # an interrupted write must not look like a valid cache
    for name in os.listdir(clients_dir):  # drop stale shards (e.g. from a run with more clients)
        if re.fullmatch(r"client_\d+", name):
            shutil.rmtree(os.path.join(clients_dir, name))

    num_classes = int(y.max()) + 1
    sizes = []
    for cid, idx in enumerate(partitions):
        idx = np.asarray(idx, dtype=np.int64)
        cdir = os.path.join(clients_dir, client_dir_name(cid))
        os.makedirs(cdir)
        np.savez(os.path.join(cdir, "train.npz"), x=x[idx], y=y[idx], indices=idx)
        counts = np.bincount(y[idx], minlength=num_classes)
        with open(os.path.join(cdir, "meta.json"), "w") as f:
            json.dump({"client_id": cid, "num_samples": int(len(idx)),
                       "num_classes_present": int((counts > 0).sum()),
                       "class_counts": counts.tolist(), **config}, f, indent=1)
        sizes.append(int(len(idx)))

    with open(cfg_path, "w") as f:  # written last: marks the partition as complete
        json.dump({"config": config, "client_sizes": sizes}, f, indent=1)
    print(f"[data] Wrote {num_clients} client folders to {clients_dir}/ "
          f"(alpha={alpha}; shard sizes min={min(sizes)} max={max(sizes)} total={sum(sizes)})")
    return sizes


def load_client_data(client_dir: str):
    """Read one client's shard from its folder -> (x uint8 [N,32,32,3], y int64 [N])."""
    with np.load(os.path.join(client_dir, "train.npz")) as d:
        return d["x"], d["y"]


# --------------------------------------------------------------------------- in-memory batching
class TensorBatches:
    """
    Iterable of (images, labels) batches served from tensors held in memory.
    train=True : reshuffles every epoch and applies RandomCrop(32, padding=4) +
                 RandomHorizontalFlip, then normalizes.
    train=False: fixed order, normalization only.
    Each instance owns its own RNG (seeded), so clients running in different
    threads never touch each other's data order / augmentation randomness.
    """

    def __init__(self, x, y, batch_size: int, train: bool, device=None, seed: int = 0):
        import torch
        self._torch = torch
        self.device = device or torch.device("cpu")
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)   # uint8 NHWC
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
        self.batch_size = batch_size
        self.train = train
        self.gen = torch.Generator().manual_seed(int(seed))
        self.mean = torch.tensor(CIFAR100_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def num_samples(self) -> int:
        return int(self.y.shape[0])

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        torch = self._torch
        n = self.num_samples
        order = torch.randperm(n, generator=self.gen) if self.train else torch.arange(n)
        order = order.to(self.device)
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            yield self._prepare(self.x[idx]), self.y[idx]

    def _prepare(self, xb):
        torch, dev = self._torch, self.device
        xb = xb.float().div_(255.0)                                   # [B,32,32,3] in [0,1]
        if self.train:
            B = xb.shape[0]
            padded = torch.nn.functional.pad(xb, (0, 0, 4, 4, 4, 4))  # zero-pad H and W by 4
            i = torch.randint(0, 9, (B,), generator=self.gen).to(dev)  # crop offsets in [0, 8]
            j = torch.randint(0, 9, (B,), generator=self.gen).to(dev)
            ar = torch.arange(32, device=dev)
            rows = (i[:, None] + ar)[:, :, None]                      # [B,32,1]
            cols = (j[:, None] + ar)[:, None, :]                      # [B,1,32]
            bidx = torch.arange(B, device=dev)[:, None, None]         # [B,1,1]
            xb = padded[bidx, rows, cols]                             # per-sample random crop -> [B,32,32,3]
            flip = (torch.rand(B, generator=self.gen) < 0.5).to(dev)
            xb = torch.where(flip[:, None, None, None], xb.flip(2), xb)  # flip along width
        xb = xb.permute(0, 3, 1, 2)                                   # -> NCHW
        return ((xb - self.mean) / self.std).contiguous()


def get_client_dataloader(client_dir: str, batch_size: int = 32, seed: int = 0, device=None):
    x, y = load_client_data(client_dir)
    return TensorBatches(x, y, batch_size, train=True, device=device, seed=seed)


def get_server_test_loader(batch_size: int = 256, data_dir: str = "data", device=None):
    x, y = load_raw_cifar100(data_dir, train=False)
    return TensorBatches(x, y, batch_size, train=False, device=device)
