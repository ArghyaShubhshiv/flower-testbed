"""Instrumented Flower server (legacy gRPC start_server, pinned to one CPU).

Per round it logs timestamps and counters for three segments:
  wait  configure_fit -> aggregate_fit   (clients training; server mostly idle)
  agg   aggregation
  eval  centralised evaluation on the test subset
plus global test accuracy/loss, so attack efficacy can be checked.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hwsig.common import single_thread_env, pin_self, mono_ns, JsonlWriter  # type: ignore
else:
    from .common import single_thread_env, pin_self, mono_ns, JsonlWriter

single_thread_env()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:8080")
    ap.add_argument("--cpu", type=int, default=-1)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n-clients", type=int, default=6)
    ap.add_argument("--agg", choices=["fedavg", "median", "trimmed_mean"], default="fedavg")
    ap.add_argument("--trim", type=int, default=2, help="per side, for trimmed_mean")
    ap.add_argument("--data", required=True)
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--model", default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", required=True)
    ap.add_argument("--run-id", default="")
    a = ap.parse_args(argv)

    pin_self(a.cpu)
    import numpy as np
    import torch
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if __package__ in (None, ""):
        from hwsig import perfctr, procstats  # type: ignore
        from hwsig.data import load_test  # type: ignore
        from hwsig.models import build_model  # type: ignore
    else:
        from . import perfctr, procstats
        from .data import load_test
        from .models import build_model

    torch.manual_seed(a.seed)
    model = build_model(a.model)
    keys = list(model.state_dict().keys())
    Xte, yte = load_test(a.data, a.test_samples)
    Xte, yte = torch.from_numpy(Xte), torch.from_numpy(yte)

    log = JsonlWriter(a.log)
    th = perfctr.CounterSet("thread")
    core = perfctr.CounterSet("core", cpu=a.cpu) if a.cpu >= 0 else None
    log.write({"event": "start", "n_params": int(sum(p.numel() for p in model.parameters()))})

    def snap():
        return (procstats.snapshot(), th.read(), core.read() if core else {}, mono_ns())

    def seg(name, s0, s1):
        d = procstats.delta(s0[0], s1[0], f"{name}.os.")
        d.update(perfctr.delta(s0[1], s1[1], f"{name}.thr."))
        if core:
            d.update(perfctr.delta(s0[2], s1[2], f"{name}.core."))
        return d

    cur: dict = {}

    def evaluate_fn(server_round, parameters, config):
        s0 = snap()
        model.load_state_dict({k: torch.from_numpy(np.asarray(v).copy())
                               for k, v in zip(keys, parameters)}, strict=True)
        model.eval()
        crit = torch.nn.CrossEntropyLoss(reduction="sum")
        loss, correct = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(yte), 500):
                out = model(Xte[i:i + 500])
                loss += float(crit(out, yte[i:i + 500]))
                correct += int((out.argmax(1) == yte[i:i + 500]).sum())
        s1 = snap()
        acc = correct / len(yte)
        rec = {"event": "round", "round": server_round,
               "test_acc": acc, "test_loss": loss / len(yte),
               "t_eval0": s0[3], "t_eval1": s1[3]}
        rec.update(seg("eval", s0, s1))
        if server_round > 0 and cur.get("round") == server_round:
            rec.update(cur["rec"])
        log.write(rec)
        return loss / len(yte), {"accuracy": acc}

    class Instrumented(FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            cur.clear()
            cur["round"] = server_round
            cur["s_cfg"] = snap()
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            s_a0 = snap()
            nds = [parameters_to_ndarrays(r.parameters) for _, r in results]
            ns = np.array([r.num_examples for _, r in results], dtype=np.float64)
            agg = []
            for layer in zip(*nds):
                S = np.stack(layer).astype(np.float64)
                if a.agg == "fedavg":
                    v = np.tensordot(ns / ns.sum(), S, axes=1)
                elif a.agg == "median":
                    v = np.median(S, axis=0)
                else:
                    k = min(a.trim, (len(S) - 1) // 2)
                    Ss = np.sort(S, axis=0)
                    v = Ss[k:len(S) - k].mean(axis=0)
                agg.append(v.astype(layer[0].dtype))
            s_a1 = snap()
            rec = {"t_cfg": cur["s_cfg"][3], "t_agg0": s_a0[3], "t_agg1": s_a1[3],
                   "n_failures": len(failures)}
            rec.update(seg("wait", cur["s_cfg"], s_a0))
            rec.update(seg("agg", s_a0, s_a1))
            cur["rec"] = rec
            return ndarrays_to_parameters(agg), {}

    init = ndarrays_to_parameters([v.detach().cpu().numpy() for v in model.state_dict().values()])
    strategy = Instrumented(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=a.n_clients, min_available_clients=a.n_clients,
        min_evaluate_clients=0,
        initial_parameters=init, evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda r: {"server_round": r},
    )
    fl.server.start_server(server_address=a.addr,
                           config=fl.server.ServerConfig(num_rounds=a.rounds),
                           strategy=strategy)
    log.write({"event": "end", "t": mono_ns()})
    log.close()


if __name__ == "__main__":
    main()
