"""Hardware/software performance counters via perf_event_open(2), from Python.

Two scopes, used side by side:

``thread`` (pid=0, cpu=-1)
    Counts only the calling thread. Training in the client runs in the main
    thread with torch.set_num_threads(1), so this is "the work this client did",
    free of anything else that ran on the core. Works at
    perf_event_paranoid <= 2 (user-space only when paranoid == 2).

``core`` (pid=-1, cpu=N)
    Counts *everything* that executes on logical CPU N: the process, its gRPC
    threads, kernel work, IRQs and anything the scheduler slipped in. Needs
    perf_event_paranoid <= 0 or CAP_PERFMON / root.

Every event is opened as its own (ungrouped) counter with
TOTAL_TIME_ENABLED|RUNNING, so when there are more events than PMCs (Zen 2 has
6 core PMCs) the kernel multiplexes and we scale: v * enabled / running.
The multiplexing ratio is recorded so the analysis can drop badly-sampled
counters.
"""

from __future__ import annotations

import ctypes
import os
import struct

from .common import read_text

_libc = ctypes.CDLL(None, use_errno=True)
_NR_perf_event_open = {"x86_64": 298, "aarch64": 241}.get(os.uname().machine, 298)

PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE = 1
PERF_TYPE_HW_CACHE = 3

PERF_FORMAT_TOTAL_TIME_ENABLED = 1
PERF_FORMAT_TOTAL_TIME_RUNNING = 2
PERF_FLAG_FD_CLOEXEC = 8

# attr.flags bit positions
_DISABLED, _INHERIT, _EXCL_USER, _EXCL_KERNEL, _EXCL_HV, _EXCL_IDLE = 0, 1, 4, 5, 6, 7

PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_RESET = 0x2403


class _Attr(ctypes.Structure):
    # PERF_ATTR_SIZE_VER1 (72 bytes) -- accepted by every kernel since 2.6.33
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
    ]


def _cache(cache_id: int, op: int, result: int) -> int:
    return cache_id | (op << 8) | (result << 16)


# name -> (type, config).  Hardware-cache ids: L1D=0 L1I=1 LL=2 DTLB=3 ITLB=4 BPU=5
# op: READ=0 WRITE=1 PREFETCH=2   result: ACCESS=0 MISS=1
EVENTS: dict[str, tuple[int, int]] = {
    # generic hardware
    "cycles": (PERF_TYPE_HARDWARE, 0),
    "instructions": (PERF_TYPE_HARDWARE, 1),
    "cache_references": (PERF_TYPE_HARDWARE, 2),
    "cache_misses": (PERF_TYPE_HARDWARE, 3),
    "branches": (PERF_TYPE_HARDWARE, 4),
    "branch_misses": (PERF_TYPE_HARDWARE, 5),
    "stalled_cycles_frontend": (PERF_TYPE_HARDWARE, 7),
    "stalled_cycles_backend": (PERF_TYPE_HARDWARE, 8),
    # hardware cache
    "l1d_loads": (PERF_TYPE_HW_CACHE, _cache(0, 0, 0)),
    "l1d_load_misses": (PERF_TYPE_HW_CACHE, _cache(0, 0, 1)),
    "l1d_prefetches": (PERF_TYPE_HW_CACHE, _cache(0, 2, 0)),
    "l1i_loads": (PERF_TYPE_HW_CACHE, _cache(1, 0, 0)),
    "l1i_load_misses": (PERF_TYPE_HW_CACHE, _cache(1, 0, 1)),
    "llc_loads": (PERF_TYPE_HW_CACHE, _cache(2, 0, 0)),
    "llc_load_misses": (PERF_TYPE_HW_CACHE, _cache(2, 0, 1)),
    "dtlb_loads": (PERF_TYPE_HW_CACHE, _cache(3, 0, 0)),
    "dtlb_load_misses": (PERF_TYPE_HW_CACHE, _cache(3, 0, 1)),
    "itlb_loads": (PERF_TYPE_HW_CACHE, _cache(4, 0, 0)),
    "itlb_load_misses": (PERF_TYPE_HW_CACHE, _cache(4, 0, 1)),
    "bpu_loads": (PERF_TYPE_HW_CACHE, _cache(5, 0, 0)),
    "bpu_load_misses": (PERF_TYPE_HW_CACHE, _cache(5, 0, 1)),
    # software (kernel-maintained, never multiplexed)
    "task_clock_ns": (PERF_TYPE_SOFTWARE, 1),
    "context_switches": (PERF_TYPE_SOFTWARE, 3),
    "cpu_migrations": (PERF_TYPE_SOFTWARE, 4),
    "minor_faults": (PERF_TYPE_SOFTWARE, 5),
    "major_faults": (PERF_TYPE_SOFTWARE, 6),
}

DEFAULT_EVENTS = list(EVENTS)


def paranoid_level() -> int:
    try:
        return int(read_text("/proc/sys/kernel/perf_event_paranoid", "4"))
    except ValueError:
        return 4


def _open(etype: int, config: int, pid: int, cpu: int, exclude_kernel: bool) -> int:
    attr = _Attr()
    attr.type = etype
    attr.size = ctypes.sizeof(_Attr)
    attr.config = config
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
    flags = (1 << _EXCL_HV)
    if exclude_kernel:
        flags |= (1 << _EXCL_KERNEL)
    attr.flags = flags  # enabled immediately (disabled bit = 0)
    fd = _libc.syscall(_NR_perf_event_open, ctypes.byref(attr),
                       ctypes.c_int(pid), ctypes.c_int(cpu),
                       ctypes.c_int(-1), ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


class CounterSet:
    """A set of independently-multiplexed counters for one scope."""

    def __init__(self, scope: str, cpu: int | None = None,
                 events: list[str] | None = None):
        self.scope = scope
        self.fds: dict[str, int] = {}
        self.errors: dict[str, str] = {}
        par = paranoid_level()
        if scope == "thread":
            pid, c = 0, -1
            exclude_kernel = par >= 2
        elif scope == "core":
            if cpu is None:
                raise ValueError("core scope needs cpu")
            pid, c = -1, int(cpu)
            exclude_kernel = False
        else:
            raise ValueError(scope)
        self.exclude_kernel = exclude_kernel
        for name in events or DEFAULT_EVENTS:
            etype, cfg = EVENTS[name]
            try:
                self.fds[name] = _open(etype, cfg, pid, c, exclude_kernel)
            except OSError as e:
                self.errors[name] = e.strerror or str(e)

    @property
    def ok(self) -> bool:
        return bool(self.fds)

    def read(self) -> dict[str, tuple[int, int, int]]:
        out = {}
        for name, fd in self.fds.items():
            try:
                buf = os.read(fd, 24)
                out[name] = struct.unpack("QQQ", buf)
            except OSError:
                pass
        return out

    def close(self) -> None:
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()


def delta(a: dict, b: dict, prefix: str = "") -> dict[str, float]:
    """Scaled deltas between two CounterSet.read() snapshots:
    count_delta * enabled_delta / running_delta per event, plus one QC value
    ``<prefix>mux_min`` = the smallest running/enabled fraction (1.0 = no
    multiplexing), emitted only when some event was actually multiplexed."""
    out: dict[str, float] = {}
    mux = 1.0
    for k, (v1, e1, r1) in b.items():
        if k not in a:
            continue
        v0, e0, r0 = a[k]
        dv, de, dr = v1 - v0, e1 - e0, r1 - r0
        if dr <= 0:
            continue
        out[f"{prefix}{k}"] = dv * (de / dr) if de else float(dv)
        if de:
            mux = min(mux, dr / de)
    if mux < 0.999:
        out[f"{prefix}mux_min"] = round(mux, 4)
    return out


def probe() -> dict:
    """What can this process measure? Used by run_one to record capabilities."""
    res = {"paranoid": paranoid_level()}
    t = CounterSet("thread")
    res["thread_ok"] = sorted(t.fds)
    res["thread_err"] = t.errors
    res["thread_exclude_kernel"] = t.exclude_kernel
    t.close()
    c = CounterSet("core", cpu=0)
    res["core_ok"] = sorted(c.fds)
    res["core_err"] = c.errors
    c.close()
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(probe(), indent=1))
