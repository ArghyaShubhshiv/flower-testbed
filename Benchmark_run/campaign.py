"""Run n honest + n ALIE runs, interleaved, randomised, thermally gated.

    python -m hwsig.campaign --name alie_v1 --pairs 10 --rounds 20

Design choices that matter for a *hardware* comparison:

* **Paired + interleaved.** Pair i uses training seed i for both conditions and
  the order inside each pair is a coin flip. Slow drift (room temperature,
  battery, background daemons) then hits both conditions equally instead of
  being aliased onto "attack".
* **Thermal gate.** Before each run wait at least --cooldown-s and until the
  hottest CPU sensor is below --cool-to-c (or --max-cool-wait-s elapses), so
  every run starts from a comparable thermal/boost state. Start/end temps are
  logged so you can include them as covariates.
* **Shadow control (default on).** In honest runs the colluders still exchange
  their update with the attacker process, which returns it untouched. Turn off
  with --no-shadow to measure the "naive" contrast where the extra process and
  IPC exist only under attack.
* **Warm-up run(s)** are executed and discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from .data import prepare
from .run_one import host_snapshot, max_temp_c, run
from .topology import describe, make_plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--pairs", type=int, default=10, help="n honest + n attack runs")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n-clients", type=int, default=6)
    ap.add_argument("--n-byz", type=int, default=2)
    ap.add_argument("--byz-ids", default="auto", help="e.g. '0,3'; auto spreads over L3 domains")
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--agg", default="fedavg", choices=["fedavg", "median", "trimmed_mean"])
    ap.add_argument("--data", default="data/cifar10")
    ap.add_argument("--max-samples", type=int, default=0, help="cap per client (0 = full shard)")
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--model", default="small", choices=["small", "medium"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--monitor-hz", type=float, default=20.0)
    ap.add_argument("--no-perf-cli", action="store_true")
    ap.add_argument("--perf-cli-events", default="")
    ap.add_argument("--cpu-plan", default="", help="override, see topology.make_plan")
    ap.add_argument("--pre-s", type=float, default=3.0)
    ap.add_argument("--post-s", type=float, default=3.0)
    ap.add_argument("--cooldown-s", type=float, default=20.0)
    ap.add_argument("--cool-to-c", type=float, default=55.0)
    ap.add_argument("--max-cool-wait-s", type=float, default=180.0)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--order-seed", type=int, default=12345)
    a = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    root = Path(a.out_root) / a.name
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(a.n_clients, a.n_byz, a.cpu_plan or None, a.byz_ids)
    print(describe(plan), flush=True)
    prepare(a.data, a.n_clients)                 # downloads/partitions only if missing

    rng = random.Random(a.order_seed)
    schedule = []
    for w in range(a.warmup_runs):
        schedule.append({"run_id": f"warmup{w}", "mode": "honest", "seed": 10_000 + w,
                         "pair": -1, "warmup": True})
    for i in range(a.pairs):
        order = ["honest", "attack"]
        rng.shuffle(order)
        for m in order:
            schedule.append({"run_id": f"p{i:03d}_{m}", "mode": m, "seed": a.seed0 + i,
                             "pair": i, "warmup": False})

    base = {k: getattr(a, k) for k in ("rounds", "n_clients", "n_byz", "z", "agg", "data",
                                        "max_samples", "model", "epochs", "batch", "lr",
                                        "monitor_hz", "pre_s", "post_s", "timeout_s",
                                        "test_samples", "perf_cli_events")}
    base["shadow"] = not a.no_shadow
    base["perf_cli"] = not a.no_perf_cli
    with open(root / "campaign.json", "w") as f:
        json.dump({"args": vars(a), "plan": plan, "host": host_snapshot(),
                   "schedule": schedule}, f, indent=1)

    for k, item in enumerate(schedule):
        rdir = root / ("_warmup" if item["warmup"] else "runs") / item["run_id"]
        if (rdir / "DONE").exists():
            print(f"[{k+1}/{len(schedule)}] {item['run_id']} already done", flush=True)
            continue
        # thermal gate
        t0 = time.time()
        time.sleep(a.cooldown_s if k else 0)
        while True:
            t = max_temp_c()
            if t is None or t <= a.cool_to_c or time.time() - t0 > a.max_cool_wait_s:
                break
            time.sleep(5)
        cfg = dict(base, **item)
        print(f"[{k+1}/{len(schedule)}] {item['run_id']}  temp={max_temp_c()}  ...", flush=True)
        t_run = time.time()
        meta = run(cfg, plan, rdir, repo)
        print(f"    -> ok={meta['ok']}  {time.time()-t_run:.0f}s  "
              f"{meta.get('error','')}", flush=True)
        if not meta["ok"]:
            print(f"    see {rdir}/logs/", file=sys.stderr, flush=True)

    # if launched via sudo, give the results back to the invoking user
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid and os.geteuid() == 0:
        for p in [root, *root.rglob("*")]:
            try:
                os.chown(p, int(uid), int(gid))
            except OSError:
                pass
    print(f"done. analyse with:  python -m hwsig.analyze {root}")


if __name__ == "__main__":
    main()
