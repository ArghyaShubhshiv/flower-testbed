# Federated Learning on CIFAR-100 (Dockerized, Flower + PyTorch)

A containerized federated learning (FL) setup: one server container and
`NUM_CLIENTS` client containers, communicating over gRPC via the
[Flower](https://flower.ai/) framework.

## How each requirement is satisfied

| # | Requirement | Implementation |
|---|---|---|
| 1 | Server & each client as separate Docker containers | `server/Dockerfile`, `client/Dockerfile`, orchestrated by `docker-compose.yml` |
| 2 | Number of clients parametrized (default 20) | `generate_compose.py --num-clients N` (default `N=20`) materializes one `client-i` service per client, since Compose YAML can't loop natively |
| 3 | CIFAR-100 distributed among clients | `common/data_utils.py`: `dirichlet_partition()` splits the CIFAR-100 **train** split across clients, guaranteeing every client gets at least `MIN_SAMPLES_PER_CLIENT` examples (see below) |
| 4 | Same model on every client | `common/model.py::CIFAR100CNN`, imported unchanged by server and client images |
| 5 | Server initializes weights, sends to clients | `server/server.py` builds a fresh `CIFAR100CNN`, wraps its weights as `initial_parameters`, and Flower ships them to every client each round via `configure_fit` |
| 6 | Same model, only data distribution varies via Dirichlet alpha | Every client trains the identical architecture; only its `Subset` (chosen by `DIRICHLET_ALPHA`) differs |
| 7 | Server reserves data clients never train on | The official CIFAR-100 **test** split (10k images) is loaded only by `server/server.py::get_evaluate_fn` and is never partitioned to clients |
| 8 | Clients send weight deltas, `local_epochs` (E) parametrized (default 5) | `client/client.py::fit()` computes `delta = new_local_weights - global_weights_received` and returns the delta; `server/strategy.py::FedAvgDelta` aggregates deltas and adds them to the tracked global weights. `LOCAL_EPOCHS` env var (default `5`) controls E |

## Architecture

```
                       ┌──────────────┐
                       │  data-init   │  downloads CIFAR-100 once into a
                       │ (runs, exits)│  shared named volume (cifar100-data)
                       └──────┬───────┘
                              │ service_completed_successfully
              ┌───────────────┼────────────────────────────┐
              ▼                                             ▼
      ┌───────────────┐                            ┌────────────────┐
      │    server      │◄──────── gRPC :8080 ─────►│   client-0..N-1  │
      │ (Flower server,│  round r: sends global W    │ (Flower clients) │
      │ FedAvgDelta     │◄──── returns delta_i ──────┤ train E epochs   │
      │ strategy,       │      + num_examples_i       │ on Dirichlet     │
      │ holds test set) │                             │ shard, alpha=α   │
      └───────────────┘                             └────────────────┘
```

Each round:
1. Server sends the current global weights `W_t` to every client (round 1 = the server's initial random weights).
2. Each client sets its local model to `W_t`, trains for `E` local epochs on its own Dirichlet-partitioned shard, then computes `Δ_i = W_i_local - W_t` and returns `(Δ_i, n_i)`.
3. Server computes the num-examples-weighted average `avg_Δ = Σ(n_i · Δ_i) / Σ(n_i)` and updates `W_{t+1} = W_t + avg_Δ` (mathematically equivalent to FedAvg over the resulting local weights, but implemented explicitly as delta transfer).
4. Server evaluates `W_{t+1}` on its held-out CIFAR-100 test set and logs loss/accuracy.

## Running it

```bash
# 1. Generate the compose file for the client count / hyperparameters you want
python3 generate_compose.py --num-clients 20 --alpha 0.5 --local-epochs 5 --num-rounds 50

# 2. Build and launch everything
docker compose up --build
```

To change the number of clients or heterogeneity later, just re-run the
generator (it overwrites `docker-compose.yml`) and re-launch:

```bash
python3 generate_compose.py --num-clients 50 --alpha 0.1 --local-epochs 3 --num-rounds 100
docker compose up --build
```

A pre-generated `docker-compose.yml` for the defaults (20 clients, alpha=0.5,
E=5, 50 rounds) is already included, so you can also just run
`docker compose up --build` directly without the generator.

## Key parameters

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `--num-clients` (generator) / `NUM_CLIENTS` | all | 20 | Number of client containers |
| `--alpha` (generator) / `DIRICHLET_ALPHA` | client | 0.5 | Dirichlet concentration. Lower = more non-IID (each client sees fewer classes); higher = closer to IID |
| `--local-epochs` (generator) / `LOCAL_EPOCHS` | server & client | 5 | Local epochs `E` a client trains per round before sending its delta |
| `--num-rounds` (generator) / `NUM_ROUNDS` | server | 50 | Total federated rounds |
| `BATCH_SIZE` | client | 32 | Local minibatch size |
| `LR` | client | 0.01 | Local SGD learning rate |
| `SEED` | all | 42 | Shared seed so every client independently derives the identical Dirichlet partition |
| `--min-samples-per-client` (generator) / `MIN_SAMPLES_PER_CLIENT` | client | 20 | Floor on each client's shard size (see below) |

## Guaranteeing every client has data

Raw per-class Dirichlet(alpha) sampling can, especially at low alpha with
many clients, leave one or more clients with very few examples or even
zero — a client can't train on nothing. `dirichlet_partition()` in
`common/data_utils.py` handles this in two steps:

1. Split each class across clients by the Dirichlet proportions as usual.
2. **Redistribute**: any client below `min_samples_per_client` (default 20)
   is topped up by moving examples from whichever client currently has the
   largest surplus above the floor, repeating until every client meets it.

This guarantees a non-empty, trainable shard for every client while still
preserving the requested heterogeneity for everyone else. It raises a
clear error at startup if `min_samples_per_client * num_clients` exceeds
the 50,000 available training images (i.e., the floor itself is
infeasible for that many clients) — lower `--min-samples-per-client` or
`--num-clients` in that case.

## Notes / things to tune for a real run

- The bundled `CIFAR100CNN` (3 conv blocks + FC) is intentionally small so the
  setup is fast to validate end-to-end; swap in a ResNet in
  `common/model.py` for stronger accuracy (all containers pick it up
  automatically since they import the same module).
- `min_fit_clients=NUM_CLIENTS` / `min_available_clients=NUM_CLIENTS` in
  `server/strategy.py`'s instantiation means the server waits for **all**
  clients to register before round 1 starts — good for reproducibility,
  but means every client container must come up successfully. Lower
  `--num-clients` for quick local smoke tests (CIFAR-100 is only 50k
  training images, so e.g. 20 clients get ~2.5k images each at alpha=0.5).
- For GPU training, add `deploy.resources.reservations.devices` (NVIDIA
  Container Toolkit) to the `server` and `client-*` blocks in
  `generate_compose.py`'s templates.
- All client containers share one Docker volume (`cifar100-data`) for the
  raw dataset so it's downloaded once by `data-init`, not by every client.
- All 20 (or however many) `client-*` services build to the **same**
  `image:` tag (`fl-cifar100-client:latest`). Docker Compose recognizes
  this and builds that image only once, then every client container just
  runs it with a different `CLIENT_ID` env var — instead of rebuilding
  (and re-downloading ~1.5GB of PyTorch/CUDA wheels) once per client,
  which is what caused very long, resource-heavy builds in earlier
  versions of this project. If you ever hand-edit the compose file,
  keep every client service's `image:` value identical for this reason.

## Troubleshooting

- **Build takes forever / Docker Desktop crashes with an `EOF` or
  "Unavailable" error during `docker compose up --build`.** This almost
  always means too many heavy image builds ran in parallel and exhausted
  the resources given to Docker Desktop (this can happen on Windows/WSL2
  in particular). Make sure you're using the shared `image:` tag setup
  described above (all client services should build the *same* image, not
  one each — check with `docker compose config | grep image:`). If it's
  still slow, increase Docker Desktop's memory/CPU allocation
  (Settings → Resources), or build first without starting anything:
  `docker compose build --parallel 1`, then `docker compose up` once that
  finishes.
- **A build hangs or fails partway through and a re-run seems stuck
  re-downloading everything.** Docker layer caching should make the second
  `docker compose build` fast (it reuses the `pip install` layer). If it
  doesn't, prune any half-finished build cache first: `docker builder
  prune` (or `docker system prune` for a more thorough clean), then
  rebuild.
