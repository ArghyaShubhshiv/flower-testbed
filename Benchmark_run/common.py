"""Small shared helpers. Import this *before* torch/numpy in every process so
thread-pool sizes are fixed before any BLAS/OpenMP runtime initialises."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "GRPC_POLL_STRATEGY": "poll",   # avoid epoll1 helper threads hopping around
    "PYTHONHASHSEED": "0",
}


def single_thread_env() -> None:
    for k, v in THREAD_ENV.items():
        os.environ.setdefault(k, v)


def pin_self(cpu: int | None) -> None:
    """Pin this process (and every thread created afterwards) to one CPU."""
    if cpu is None or cpu < 0:
        return
    os.sched_setaffinity(0, {int(cpu)})


def mono_ns() -> int:
    """CLOCK_MONOTONIC in ns: identical clock across all processes on the box,
    so client/server/monitor timestamps can be joined directly."""
    return time.monotonic_ns()


class JsonlWriter:
    def __init__(self, path: str | os.PathLike):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", buffering=1)

    def write(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps(rec, default=_jsonable) + "\n")

    def close(self) -> None:
        self._f.close()


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(o)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def read_text(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def prefixed(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def chunks(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# --------------------------------------------------------------------------- #
# log compaction: keep only what changes
# --------------------------------------------------------------------------- #
DATA_EVENTS = ("fit", "round")


def _round_sig(v, sig: int = 7):
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        if v.is_integer() and abs(v) < 2 ** 53:
            return int(v)
        return float(f"{v:.{sig}g}")
    return v


def compact_jsonl(path: str | os.PathLike) -> None:
    """Rewrite a JSONL log so per-round records hold only values that change.

    Keys whose value is identical in every data record (or always null) are
    moved once into ``<stem>.const.json``; read back with ``read_records``."""
    p = Path(path)
    recs = read_jsonl(p)
    data = [r for r in recs if r.get("event") in DATA_EVENTS]
    other = [r for r in recs if r.get("event") not in DATA_EVENTS and r.get("event") != "end"]
    if not data:
        return
    data = [{k: _round_sig(v) for k, v in r.items() if v is not None} for r in data]
    keys = set().union(*data)
    const = {}
    if len(data) > 1:
        for k in keys:
            if k in ("event", "round"):
                continue
            vals = [r.get(k) for r in data]
            if all(v == vals[0] for v in vals):
                const[k] = vals[0]
    for r in data:
        for k in const:
            r.pop(k, None)
    for r in other:
        const.setdefault("_" + str(r.get("event")), {k: v for k, v in r.items() if k != "event"})
    with open(p, "w") as f:
        for r in data:
            f.write(json.dumps(r, default=_jsonable, separators=(",", ":")) + "\n")
    with open(p.with_suffix(".const.json"), "w") as f:
        json.dump(const, f, default=_jsonable, separators=(",", ":"))


def read_records(path: str | os.PathLike, event: str) -> list[dict]:
    """Data records of one event type with the run's constants merged back in."""
    p = Path(path)
    cp = p.with_suffix(".const.json")
    const = {}
    if cp.exists():
        const = {k: v for k, v in json.loads(cp.read_text()).items() if not k.startswith("_")}
    return [{**const, **r} for r in read_jsonl(p) if r.get("event") == event]


def compact_csv(path: str | os.PathLike, meta_path: str | os.PathLike) -> None:
    """Drop columns that never change; record their single value in meta['constants']."""
    import csv
    p = Path(path)
    if not p.exists():
        return
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return
    head, body = rows[0], rows[1:]
    body = [r for r in body if len(r) == len(head)]
    keep, const = [], {}
    for j, h in enumerate(head):
        col = [r[j] for r in body]
        if j == 0 or any(v != col[0] for v in col):
            keep.append(j)
        elif col[0] != "":
            const[h] = col[0]
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([head[j] for j in keep])
        for r in body:
            w.writerow([r[j] for j in keep])
    mp = Path(meta_path)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    meta["constants"] = const
    meta["n_columns_kept"] = len(keep)
    mp.write_text(json.dumps(meta, indent=1))


def compact_run(rdir: str | os.PathLike) -> None:
    rdir = Path(rdir)
    for p in rdir.glob("*.jsonl"):
        compact_jsonl(p)
    compact_csv(rdir / "monitor.csv", rdir / "monitor_meta.json")
