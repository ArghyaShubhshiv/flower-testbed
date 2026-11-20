"""In-process OS / runtime counters (no privileges needed).

Deliberately non-redundant: each quantity is read from exactly one source.
``snapshot()`` returns a flat dict; ``delta(a, b)`` subtracts cumulative
counters and keeps the end value of gauges.
"""

from __future__ import annotations

import gc
import resource
import time

RUSAGE_THREAD = getattr(resource, "RUSAGE_THREAD", 1)

GAUGES = {"rss_kB", "hwm_kB", "threads"}

# /proc/thread-self/sched keys (names differ across kernel versions)
_SCHED = {
    "se.nr_migrations": "migrations",
    "se.statistics.wait_sum": "wait_sum_ms", "wait_sum": "wait_sum_ms",
    "se.statistics.wait_count": "wait_count", "wait_count": "wait_count",
    "se.statistics.iowait_sum": "iowait_sum_ms", "iowait_sum": "iowait_sum_ms",
    "se.statistics.nr_wakeups": "wakeups", "nr_wakeups": "wakeups",
}


def _kv(path: str) -> dict:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(":")
                if _:
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def snapshot() -> dict:
    s: dict[str, float] = {"mono_ns": time.monotonic_ns(),
                           "thread_cpu_ns": time.thread_time_ns(),
                           "proc_cpu_ns": time.process_time_ns()}
    try:
        t = resource.getrusage(RUSAGE_THREAD)          # the training thread
        s.update(thr_utime_s=t.ru_utime, thr_stime_s=t.ru_stime,
                 thr_minflt=t.ru_minflt, thr_majflt=t.ru_majflt,
                 thr_vol_cs=t.ru_nvcsw, thr_invol_cs=t.ru_nivcsw)
    except (ValueError, OSError):
        pass
    p = resource.getrusage(resource.RUSAGE_SELF)       # whole process incl. gRPC threads
    s.update(proc_minflt=p.ru_minflt, proc_majflt=p.ru_majflt,
             proc_vol_cs=p.ru_nvcsw, proc_invol_cs=p.ru_nivcsw,
             proc_blk_in=p.ru_inblock, proc_blk_out=p.ru_oublock)

    io = _kv("/proc/self/io")
    for k in ("rchar", "wchar", "syscr", "syscw"):
        if k in io:
            s[f"io_{k}"] = int(io[k])
    st = _kv("/proc/self/status")
    if "VmRSS" in st:
        s["rss_kB"] = int(st["VmRSS"].split()[0])
    if "VmHWM" in st:
        s["hwm_kB"] = int(st["VmHWM"].split()[0])
    if "Threads" in st:
        s["threads"] = int(st["Threads"])
    try:
        with open("/proc/self/schedstat") as f:
            run, wait, slices = f.read().split()[:3]
        s.update(rq_run_ns=int(run), rq_wait_ns=int(wait), timeslices=int(slices))
    except (OSError, ValueError):
        pass
    for k, v in _kv("/proc/thread-self/sched").items():
        if k in _SCHED:
            try:
                s[_SCHED[k]] = float(v)
            except ValueError:
                pass
    for i, g in enumerate(gc.get_stats()):
        s[f"gc{i}"] = g.get("collections", 0)
    return s


def delta(a: dict, b: dict, prefix: str = "") -> dict:
    out = {}
    for k, vb in b.items():
        if k == "mono_ns":
            continue
        if k in GAUGES:
            out[f"{prefix}{k}"] = vb
        elif k in a:
            out[f"{prefix}{k}"] = vb - a[k]
    if "mono_ns" in a and "mono_ns" in b:
        out[f"{prefix}wall_ns"] = b["mono_ns"] - a["mono_ns"]
    return out
