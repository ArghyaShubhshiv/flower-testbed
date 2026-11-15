"""
CIFAR-100 loading + Dirichlet-based non-IID partitioning across clients.

Partitioning strategy (Hsu et al., 2019 style):
  For each class c, draw a proportion vector p ~ Dirichlet(alpha, ..., alpha)
  over the N clients, and split that class's examples across clients
  according to p. A single seed is shared by every client/server process so
  they all independently derive the *same* partition without any shared
  state file -- client i simply keeps partition[i].

  alpha -> 0   : each client sees (close to) a single class -> highly non-IID
  alpha -> inf : partition approaches uniform/IID across clients

The official CIFAR-100 test split (10k images) is NEVER partitioned to
clients. It is reserved exclusively for centralized evaluation on the
server (requirement: server holds out data clients never train on).
"""
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def load_full_trainset(data_dir: str = "/data"):
    return datasets.CIFAR100(root=data_dir, train=True, download=True, transform=get_transforms(True))


def load_full_testset(data_dir: str = "/data"):
    return datasets.CIFAR100(root=data_dir, train=False, download=True, transform=get_transforms(False))


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


def get_client_dataset(client_id: int, num_clients: int, alpha: float, data_dir: str = "/data", seed: int = 42,
                        min_samples_per_client: int = 20):
    trainset = load_full_trainset(data_dir)
    partitions = dirichlet_partition(trainset.targets, num_clients, alpha, seed, min_samples_per_client)
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id {client_id} out of range for num_clients={num_clients}")
    return Subset(trainset, partitions[client_id])


def get_client_dataloader(client_id: int, num_clients: int, alpha: float, batch_size: int = 32,
                           data_dir: str = "/data", seed: int = 42, min_samples_per_client: int = 20):
    ds = get_client_dataset(client_id, num_clients, alpha, data_dir, seed, min_samples_per_client)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)


def get_server_test_loader(batch_size: int = 256, data_dir: str = "/data"):
    testset = load_full_testset(data_dir)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
