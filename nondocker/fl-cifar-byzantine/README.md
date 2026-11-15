# Federated Learning on CIFAR-100 (Flower + PyTorch, single process, threaded clients)

One launcher script, `run_simulation.py`, runs the whole federation inside a single
Python process: a Flower server in one thread and `NUM_CLIENTS` Flower clients in
their own threads, talking over loopback gRPC. No Docker, no per-client images,
no per-client dataset loading.

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.8-3.11 (flwr 1.8.0 does not support 3.12)
pip install -r requirements.txt
```

## Run

```bash
python run_simulation.py                                   # 20 clients, alpha=0.5, E=5, 50 rounds
python run_simulation.py --num-clients 10 --num-rounds 20 --local-epochs 2
python run_simulation.py --attack sign_flip --num-byzantine 4
python run_simulation.py --partition-only                  # only (re)create the client folders
python analysis/compare_runs.py --plot                     # compare results/*.csv against baseline.csv
```

On first run the script downloads CIFAR-100 into `data/` (if it isn't there already) and
writes the client folders. Later runs reuse them as long as `--num-clients`, `--alpha`,
`--seed` and `--min-samples-per-client` are unchanged, so baseline-vs-attack comparisons
train on exactly the same shards.

## Layout

```
run_simulation.py        launcher: partition -> start server thread -> start client threads
common/
  model.py               CIFAR100CNN + get/set_parameters helpers (shared by server & clients)
  data_utils.py          Dirichlet partitioning, per-client folder writer, in-memory batching
  attacks.py             Byzantine attacks (unchanged)
client/client.py         DeltaClient (Flower NumPyClient, returns weight deltas)
server/server.py         server thread entry point + centralized evaluation
server/strategy.py       FedAvgDelta (unchanged)
analysis/compare_runs.py
data/                    raw CIFAR-100 (downloaded once)
client_data/             GENERATED: one folder per client
  partition_config.json    config + shard sizes (used to decide whether to reuse)
  client_0/train.npz       x: uint8 [N,32,32,3], y: int64 [N], indices: ids into the CIFAR-100 train split
  client_0/meta.json       shard size, number of classes present, per-class counts
  client_1/ ...
results/                 <run-name>.csv per run
```

## What stayed the same

| Requirement | Implementation |
|---|---|
| Number of clients parametrized (default 20) | `--num-clients` |
| CIFAR-100 distributed among clients | `common/data_utils.py::dirichlet_partition()` -- **unchanged**, same seed gives the same indices as the Docker version |
| Every client gets at least `MIN_SAMPLES` | same top-up-from-surplus redistribution (`--min-samples-per-client`, default 20) |
| Same model everywhere | `common/model.py::CIFAR100CNN` |
| Server initializes weights and sends them out | `run_simulation.py` builds the seeded initial weights; passed as `initial_parameters` to the strategy |
| Server holds out data clients never train on | official CIFAR-100 **test** split, loaded only by `server/server.py`, never partitioned |
| Clients send deltas, `E` parametrized (default 5) | `DeltaClient.fit()` returns `new - received`; `FedAvgDelta` aggregates and adds to the global weights; `--local-epochs` |
| Byzantine attacks | `--attack`, `--num-byzantine`, `--attack-scale`, `--attack-start-round`; logic in `common/attacks.py` is unchanged |

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--num-clients` | 20 | Number of client threads |
| `--alpha` | 0.5 | Dirichlet concentration; lower = more non-IID |
| `--local-epochs` | 5 | Local epochs per round |
| `--num-rounds` | 50 | Federated rounds |
| `--batch-size` / `--lr` | 32 / 0.01 | Local SGD settings (momentum 0.9) |
| `--fraction-fit` | 1.0 | Fraction of clients sampled per round |
| `--seed` | 42 | Seeds the partition, initial weights, and each client's batching/augmentation |
| `--min-samples-per-client` | 20 | Floor on shard size (raises an error if infeasible for that many clients) |
| `--attack`, `--num-byzantine`, `--attack-scale`, `--attack-start-round`, `--run-name` | none / 0 / per-attack / 1 / auto | Byzantine setup; clients `0..f-1` are Byzantine |
| `--data-dir`, `--clients-dir`, `--results-dir` | `data`, `client_data`, `results` | Paths |
| `--force-repartition` | off | Rebuild client folders even if up to date |
| `--max-parallel-clients` | all on GPU, else `min(num_clients, cores)` | How many clients train at the same time |
| `--torch-threads` | `cores // max-parallel-clients` | Torch intra-op threads per training client |
| `--cpu` | off | Force CPU |
| `--verbose-clients` | off | Log every local epoch of every client |

## Performance notes

- All client threads connect to the server (it still waits for all of them before round 1),
  but a semaphore lets only `--max-parallel-clients` of them train at once, and each gets
  `cores // max-parallel` torch threads, so the CPU isn't oversubscribed. If you have a GPU,
  everything runs on it and all clients train concurrently by default.
- Data is held in memory as tensors per client; random-crop/flip/normalize is done per batch.
  There are no DataLoader worker processes. (Same augmentation as before: RandomCrop(32, padding=4),
  horizontal flip, CIFAR-100 mean/std normalization.)
- `client_data/` is about 150 MB (uncompressed uint8 arrays, so loading is fast). It is safe to delete;
  it is rebuilt automatically.

## Reproducibility caveat

The partition, initial weights, and each client's data order/augmentation are seeded.
Dropout and BatchNorm still use PyTorch's global RNG, which is shared by threads, so two runs with the
same seed will be close but not bit-identical when several clients train in parallel. For exact
repeatability use `--max-parallel-clients 1`.

## Notes

- `min_fit_clients=NUM_CLIENTS` / `min_available_clients=NUM_CLIENTS`: the server waits for all
  clients to register before round 1. If any client thread crashes, the launcher prints its traceback
  and exits instead of hanging.
- The bundled CNN is intentionally small; swap in a ResNet in `common/model.py` and both server and
  clients pick it up.
