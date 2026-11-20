"""Instrumented Flower client: one OS process, pinned to one CPU, one thread of
torch compute.

Every fit() is split into segments, each with OS counters (procstats) and
perf counters for this thread (``thr``) and for the whole core (``core``):

  idle   end of previous fit -> start of this fit   (waiting for the server,
                                                     Flower (de)serialisation)
  train  local SGD
  post   flatten update, [colluders only] ship it to the attacker and wait for
         the reply, rebuild parameters

Model-level quantities (loss, update norm, cosine own-vs-submitted) are logged
under ``model.*`` so the analysis can compare the hardware signal with the
classical "look at the update" baseline, and keep the two apart.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hwsig.common import single_thread_env, pin_self, mono_ns, JsonlWriter  # type: ignore
else:
    from .common import single_thread_env, pin_self, mono_ns, JsonlWriter

single_thread_env()


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--cpu", type=int, default=-1)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-clients", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--model", default="small")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--colluder", action="store_true",
                    help="this client is one of the designated Byzantine ids")
    ap.add_argument("--attacker-addr", default="")
    ap.add_argument("--authkey", default="hwsig")
    ap.add_argument("--log", required=True)
    ap.add_argument("--run-id", default="")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    pin_self(a.cpu)

    import numpy as np
    import torch
    from torch import nn
    import flwr as fl

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if __package__ in (None, ""):
        from hwsig import perfctr, procstats  # type: ignore
        from hwsig.data import load_client  # type: ignore
        from hwsig.models import build_model  # type: ignore
    else:
        from . import perfctr, procstats
        from .data import load_client
        from .models import build_model

    log = JsonlWriter(a.log)
    X, y = load_client(a.data, a.n_clients, a.cid, a.max_samples)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    model = build_model(a.model)
    keys = list(model.state_dict().keys())

    th = perfctr.CounterSet("thread")
    core = perfctr.CounterSet("core", cpu=a.cpu) if a.cpu >= 0 else None

    conn = None
    if a.colluder:
        from multiprocessing.connection import Client
        for _ in range(600):
            try:
                conn = Client(a.attacker_addr, family="AF_UNIX", authkey=a.authkey.encode())
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.1)
        if conn is None:
            raise SystemExit("cannot reach attacker")
        conn.send({"cid": a.cid})

    log.write({"event": "start", "cid": a.cid, "cpu": a.cpu, "n_samples": int(len(y)),
               "affinity": sorted(os.sched_getaffinity(0))})

    def snap():
        return (procstats.snapshot(), th.read(), core.read() if core else {}, mono_ns())

    def seg(name, s0, s1):
        d = procstats.delta(s0[0], s1[0], f"{name}.os.")
        d.update(perfctr.delta(s0[1], s1[1], f"{name}.thr."))
        if core:
            d.update(perfctr.delta(s0[2], s1[2], f"{name}.core."))
        return d

    def flat(nds):
        return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in nds])

    def unflat(vec, like):
        out, o = [], 0
        for t in like:
            n = t.size
            out.append(vec[o:o + n].reshape(t.shape).astype(t.dtype, copy=False))
            o += n
        return out

    state = {"prev_end": snap()}

    class C(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [v.detach().cpu().numpy() for v in model.state_dict().values()]

        def fit(self, parameters, config):
            rnd = int(config.get("server_round", 0))
            s0 = snap()
            model.load_state_dict({k: torch.from_numpy(np.asarray(v).copy())
                                   for k, v in zip(keys, parameters)}, strict=True)
            g = torch.Generator().manual_seed(a.seed * 100_003 + a.cid * 1009 + rnd)
            torch.manual_seed(a.seed * 100_003 + a.cid * 1009 + rnd)
            opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
            crit = nn.CrossEntropyLoss()
            model.train()
            tot_loss, tot_n, correct, nb = 0.0, 0, 0, 0
            for _ in range(a.epochs):
                perm = torch.randperm(len(y), generator=g)
                for i in range(0, len(y), a.batch):
                    idx = perm[i:i + a.batch]
                    xb, yb = X[idx], y[idx]
                    opt.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
                    tot_loss += float(loss) * len(yb)
                    tot_n += len(yb)
                    correct += int((out.argmax(1) == yb).sum())
                    nb += 1
            s1 = snap()

            new = [v.detach().cpu().numpy() for v in model.state_dict().values()]
            glob_flat = flat(parameters)
            own = flat(new) - glob_flat
            submitted = own
            t_tx = t_rx = None
            if conn is not None:
                t_tx = mono_ns()
                conn.send({"round": rnd, "cid": a.cid, "n": int(own.size)})
                conn.send_bytes(own.tobytes())
                submitted = np.frombuffer(conn.recv_bytes(), dtype=np.float32)
                t_rx = mono_ns()
            out_params = unflat(glob_flat + submitted, new)
            s2 = snap()

            n_own = float(np.linalg.norm(own))
            n_sub = float(np.linalg.norm(submitted))
            cos = float(own @ submitted / (n_own * n_sub + 1e-12))
            rec = {"event": "fit", "cid": a.cid, "round": rnd,
                   "t_idle0": state["prev_end"][3], "t_fit0": s0[3], "t_train1": s1[3],
                   "t_fit1": s2[3], "t_tx": t_tx, "t_rx": t_rx,
                   "model.train_loss": tot_loss / max(tot_n, 1),
                   "model.train_acc": correct / max(tot_n, 1),
                   "model.own_update_norm": n_own,
                   "model.submitted_update_norm": n_sub,
                   "model.cos_own_submitted": cos,
                   "model.global_norm": float(np.linalg.norm(glob_flat))}
            rec.update(seg("idle", state["prev_end"], s0))
            rec.update(seg("train", s0, s1))
            rec.update(seg("post", s1, s2))
            log.write(rec)
            state["prev_end"] = snap()
            return out_params, int(len(y)), {"cid": a.cid}

        def evaluate(self, parameters, config):
            return 0.0, 1, {}

    fl.client.start_client(server_address=a.server, client=C().to_client(), insecure=True)
    log.write({"event": "end", "t": mono_ns()})
    log.close()
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()
