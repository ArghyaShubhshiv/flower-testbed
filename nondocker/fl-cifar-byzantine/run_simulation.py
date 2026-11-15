#!/usr/bin/env python3
"""
Single-process federated learning launcher (no Docker).

What it does:
  1. Downloads CIFAR-100 if needed and partitions the TRAIN split with the same
     Dirichlet logic as before, writing one folder per client:
         client_data/client_<i>/train.npz  (+ meta.json)
     (re-used automatically on later runs with the same num-clients/alpha/seed/min-samples)
  2. Starts the Flower server in a background thread (loopback gRPC, free port).
  3. Starts one thread per client; each thread loads ONLY its own folder and
     runs a standard Flower NumPyClient against the server.
  4. Waits for all rounds to finish; results go to results/<run-name>.csv.

Examples:
  python run_simulation.py                                  # 20 clients, alpha=0.5, E=5, 50 rounds
  python run_simulation.py --num-clients 10 --num-rounds 20 --local-epochs 2
  python run_simulation.py --attack sign_flip --num-byzantine 4
  python run_simulation.py --partition-only                 # just (re)create the client folders
"""
# Must be set before flwr is imported: avoids Flower's startup telemetry network call.
import os
os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")

import argparse
import socket
import sys
import threading
import time

import numpy as np
import torch

from common import attacks
from common.data_utils import client_dir_name, partition_to_folders
from common.model import CIFAR100CNN, get_parameters


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # federation / training
    p.add_argument("--num-clients", type=int, default=20, help="Number of client threads (default: 20)")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha controlling data heterogeneity")
    p.add_argument("--local-epochs", type=int, default=5, help="Local epochs E per client per round (default: 5)")
    p.add_argument("--num-rounds", type=int, default=50, help="Number of federated rounds")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled per round")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-samples-per-client", type=int, default=20,
                   help="Guaranteed minimum training examples per client (topped up from surplus clients)")
    # byzantine
    p.add_argument("--attack", default="none", choices=attacks.ATTACK_NAMES, help="Byzantine attack (default: none)")
    p.add_argument("--num-byzantine", type=int, default=0, help="Byzantine clients (ids 0..f-1)")
    p.add_argument("--attack-scale", type=float, default=None, help="Attack strength (per-attack default)")
    p.add_argument("--attack-start-round", type=int, default=1)
    p.add_argument("--run-name", default=None, help="Results file name (default: baseline / <attack>_f<f>)")
    # paths
    p.add_argument("--data-dir", default="data", help="Where raw CIFAR-100 lives / is downloaded to")
    p.add_argument("--clients-dir", default="client_data", help="Where the per-client folders are written")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--force-repartition", action="store_true", help="Rebuild client folders even if up to date")
    p.add_argument("--partition-only", action="store_true", help="Create the client folders and exit")
    # performance
    p.add_argument("--max-parallel-clients", type=int, default=None,
                   help="Max clients training simultaneously (default: all on GPU, else min(num_clients, CPU cores)). "
                        "All client threads still exist; the rest wait their turn.")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Intra-op torch threads per training client (default: cores // max-parallel-clients)")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--verbose-clients", action="store_true", help="Log every local epoch of every client")

    args = p.parse_args()
    if args.attack != "none" and not 0 < args.num_byzantine < args.num_clients:
        p.error("--attack requires 0 < --num-byzantine < --num-clients")
    if args.attack == "none":
        args.num_byzantine = 0
    if args.attack_scale is None:
        args.attack_scale = attacks.DEFAULT_SCALE.get(args.attack, 1.0)
    args.run_name = args.run_name or ("baseline" if args.attack == "none"
                                      else f"{args.attack}_f{args.num_byzantine}")
    return args


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, server_thread: threading.Thread, server_out: dict, timeout: float = 180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("Server thread exited before accepting connections:\n"
                               + (server_out.get("error", (None, None, "no error recorded"))[2]))
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Flower server did not start listening on port {port} within {timeout:.0f}s")


def main():
    args = parse_args()

    # 1. Data: partition CIFAR-100 train split into per-client folders.
    partition_to_folders(args.data_dir, args.clients_dir, args.num_clients, args.alpha,
                         args.seed, args.min_samples_per_client, force=args.force_repartition)
    if args.partition_only:
        return 0

    # Imported here (not at the top) so --partition-only / --help never need flwr.
    from client.client import DeltaClient, run_client
    from server.server import run_server

    attack = attacks.AttackConfig(name=args.attack, num_byzantine=args.num_byzantine,
                                  scale=args.attack_scale, start_round=args.attack_start_round)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cores = os.cpu_count() or 1
    max_parallel = args.max_parallel_clients or (args.num_clients if device.type == "cuda"
                                                 else min(args.num_clients, cores))
    max_parallel = max(1, min(max_parallel, args.num_clients))
    torch_threads = args.torch_threads or max(1, cores // max_parallel)
    train_slots = threading.BoundedSemaphore(max_parallel)
    print(f"[sim] device={device} cores={cores} | {args.num_clients} client threads, "
          f"{max_parallel} training at a time, {torch_threads} torch thread(s) each", flush=True)

    # 2. Server-initialized global model (seeded; built here, before any client exists).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    initial_weights = get_parameters(CIFAR100CNN())

    # 3. Server thread (loopback gRPC on a free port).
    port = free_port()
    address = f"127.0.0.1:{port}"
    errors: list = []
    server_out: dict = {}
    server_thread = threading.Thread(
        target=run_server, name="fl-server", daemon=True,
        kwargs=dict(server_address=address, initial_weights=initial_weights, num_rounds=args.num_rounds,
                    num_clients=args.num_clients, local_epochs=args.local_epochs,
                    fraction_fit=args.fraction_fit, attack=attack, results_dir=args.results_dir,
                    run_name=args.run_name, data_dir=args.data_dir, device=device,
                    torch_threads=torch_threads, out=server_out))
    start = time.time()
    server_thread.start()
    wait_for_port(port, server_thread, server_out)

    # 4. Client threads: each loads only its own client_data/client_<i>/ folder.
    clients = [DeltaClient(cid, os.path.join(args.clients_dir, client_dir_name(cid)), attack,
                           local_epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr,
                           seed=args.seed, device=device, torch_threads=torch_threads,
                           train_slots=train_slots, verbose=args.verbose_clients)
               for cid in range(args.num_clients)]
    client_threads = [threading.Thread(target=run_client, args=(c, address, errors),
                                       name=f"client-{c.client_id}", daemon=True) for c in clients]
    for t in client_threads:
        t.start()

    # 5. Wait for the run; fail fast if any thread crashes (otherwise the server would wait forever).
    try:
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
            if errors or "error" in server_out:
                break
    except KeyboardInterrupt:
        print("\n[sim] Interrupted.", flush=True)
        os._exit(130)

    failures = errors + ([server_out["error"]] if "error" in server_out else [])
    if failures:
        for who, _, tb in failures:
            print(f"\n[sim] {who} crashed:\n{tb}", file=sys.stderr, flush=True)
        os._exit(1)

    for t in client_threads:
        t.join(timeout=30)

    elapsed = time.time() - start
    hist = server_out.get("history")
    accs = hist.metrics_centralized.get("accuracy", []) if hist else []
    print(f"\n[sim] Finished {args.num_rounds} rounds in {elapsed / 60:.1f} min"
          + (f" | final test accuracy = {accs[-1][1]:.4f}" if accs else "")
          + f" | results: {os.path.join(args.results_dir, args.run_name)}.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
