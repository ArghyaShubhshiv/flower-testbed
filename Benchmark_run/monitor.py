"""System-wide sampler, run as its own process on the housekeeping CPU.

Samples, at ``--hz`` (default 20 Hz), every cumulative counter and gauge the
kernel exposes per CPU and per package, and writes one CSV row per sample with
a CLOCK_MONOTONIC timestamp so it can be joined with the client/server logs.

Sources (each silently skipped if absent or unreadable):

  /proc/stat                 per-CPU user/nice/system/idle/iowait/irq/softirq/steal
                             jiffies; system ctxt, intr, forks, procs_running/blocked
  /sys/.../cpufreq           per-CPU scaling_cur_freq (kHz)
  /sys/.../cpuidle           per-CPU per-C-state entry count and residency (us)
  /proc/interrupts           per-CPU device-IRQ total + named rows (LOC, RES, CAL,
                             TLB, NMI, PMI, IWI, ...): timer ticks, resched IPIs,
                             function-call IPIs, TLB shootdowns
  /proc/softirqs             per-CPU TIMER, NET_RX, SCHED, RCU, ...
  /proc/schedstat            per-CPU run time, runqueue wait time, timeslices
  /sys/class/hwmon           k10temp Tctl/Tccd, amdgpu, nvme, fans, voltages,
                             power, current, and amd_energy per-core energy if loaded
  /sys/class/powercap        RAPL energy_uj (package/core) -- root on >= 5.10
  /dev/cpu/N/msr             AMD only: CORE_ENERGY_STAT (0xC001029A) per physical
                             core and PKG_ENERGY_STAT (0xC001029B) -- root + msr module
  /sys/class/power_supply    battery power_now / current_now / voltage_now, AC online
  /proc/meminfo, /proc/vmstat, /proc/pressure/*, /proc/net/dev (lo), /proc/net/snmp
                             (TCP segments), /proc/loadavg

Optionally also runs ``perf stat -A -I <ms> -x,`` for named PMU events
(APERF/MPERF, RAPL, system-wide core events, plus any vendor-specific events
passed with --perf-cli-events) into perf_cli.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hwsig.common import mono_ns, pin_self, read_text  # type: ignore
    from hwsig.topology import logical_cpus, physical_cores  # type: ignore
else:
    from .common import mono_ns, pin_self, read_text
    from .topology import logical_cpus, physical_cores

_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


class PFile:
    """A procfs/sysfs file kept open and re-read with pread(offset=0)."""

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)

    def read(self) -> str:
        chunks, off = [], 0
        while True:
            b = os.pread(self.fd, 65536, off)
            if not b:
                break
            chunks.append(b)
            off += len(b)
            if len(b) < 65536:
                break
        return b"".join(chunks).decode(errors="replace")


def try_pfile(path: str) -> PFile | None:
    try:
        f = PFile(path)
        f.read()
        return f
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# individual sources: each has .cols (list[str]) and .read() -> list[number]
# --------------------------------------------------------------------------- #
class ProcStat:
    FIELDS = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]

    def __init__(self, cpus):
        self.f = PFile("/proc/stat")
        self.cpus = cpus
        self.cols = [f"cpu{c}_stat_{x}" for c in cpus for x in self.FIELDS]
        self.cols += ["sys_ctxt", "sys_intr", "sys_forks", "sys_procs_running", "sys_procs_blocked"]

    def read(self):
        per, sysv = {}, {}
        for line in self.f.read().splitlines():
            p = line.split()
            if not p:
                continue
            if p[0].startswith("cpu") and p[0] != "cpu":
                per[int(p[0][3:])] = [int(x) for x in p[1:9]]
            elif p[0] == "ctxt":
                sysv["ctxt"] = int(p[1])
            elif p[0] == "intr":
                sysv["intr"] = int(p[1])
            elif p[0] == "processes":
                sysv["forks"] = int(p[1])
            elif p[0] == "procs_running":
                sysv["procs_running"] = int(p[1])
            elif p[0] == "procs_blocked":
                sysv["procs_blocked"] = int(p[1])
        out = []
        for c in self.cpus:
            v = per.get(c, [None] * 8)
            out.extend((v + [None] * 8)[:8])
        out += [sysv.get(k) for k in ("ctxt", "intr", "forks", "procs_running", "procs_blocked")]
        return out


class MultiFile:
    """Many single-value sysfs files -> one column each."""

    def __init__(self, items: list[tuple[str, str]], scale: float = 1.0):
        self.files, self.cols = [], []
        for col, path in items:
            f = try_pfile(path)
            if f is not None:
                self.files.append(f)
                self.cols.append(col)

    def read(self):
        out = []
        for f in self.files:
            try:
                out.append(int(f.read().strip().split()[0]))
            except (ValueError, IndexError, OSError):
                out.append(None)
        return out


def cpufreq_source(cpus):
    return MultiFile([(f"cpu{c}_freq_khz",
                       f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq") for c in cpus])


def cpuidle_source(cpus, meta):
    items = []
    names = {}
    for c in cpus:
        for sd in sorted(glob.glob(f"/sys/devices/system/cpu/cpu{c}/cpuidle/state*")):
            k = sd.rsplit("state", 1)[1]
            names[k] = read_text(f"{sd}/name", k)
            items.append((f"cpu{c}_cstate{k}_usage", f"{sd}/usage"))
            items.append((f"cpu{c}_cstate{k}_time_us", f"{sd}/time"))
    meta["cstate_names"] = names
    return MultiFile(items)


class TableByCPU:
    """/proc/interrupts and /proc/softirqs style tables."""

    def __init__(self, path: str, cpus, prefix: str, numeric_total: bool):
        self.f = PFile(path)
        self.cpus = cpus
        self.prefix = prefix
        self.numeric_total = numeric_total
        header, rows = self._parse()
        self.header = header
        self.rownames = sorted(r for r in rows if r != "__dev_total__")
        self.cols = []
        for c in cpus:
            if numeric_total:
                self.cols.append(f"cpu{c}_{prefix}_dev_total")
            for r in self.rownames:
                self.cols.append(f"cpu{c}_{prefix}_{r}")

    def _parse(self):
        lines = self.f.read().splitlines()
        header = [int(h[3:]) for h in lines[0].split() if h.startswith("CPU")]
        rows: dict[str, list[int]] = {}
        dev_total = [0] * len(header)
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            name = name.strip()
            vals = rest.split()
            nums = []
            for v in vals[:len(header)]:
                try:
                    nums.append(int(v))
                except ValueError:
                    break
            if len(nums) < len(header):
                continue
            if name.isdigit():
                dev_total = [a + b for a, b in zip(dev_total, nums)]
            else:
                rows[name] = nums
        if self.numeric_total:
            rows["__dev_total__"] = dev_total
        return header, rows

    def read(self):
        header, rows = self._parse()
        idx = {c: i for i, c in enumerate(header)}
        out = []
        for c in self.cpus:
            i = idx.get(c)
            if self.numeric_total:
                out.append(rows["__dev_total__"][i] if i is not None else None)
            for r in self.rownames:
                v = rows.get(r)
                out.append(v[i] if (v is not None and i is not None) else None)
        return out


class SchedStat:
    def __init__(self, cpus):
        self.f = PFile("/proc/schedstat")
        self.cpus = cpus
        self.cols = [f"cpu{c}_sched_{x}" for c in cpus for x in ("run_ns", "wait_ns", "slices")]

    def read(self):
        d = {}
        for line in self.f.read().splitlines():
            p = line.split()
            if p and p[0].startswith("cpu") and p[0] != "cpu":
                try:
                    d[int(p[0][3:])] = [int(p[7]), int(p[8]), int(p[9])]
                except (IndexError, ValueError):
                    pass
        out = []
        for c in self.cpus:
            out.extend(d.get(c, [None, None, None]))
        return out


def hwmon_source(meta):
    items = []
    info = {}
    for hd in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read_text(f"{hd}/name", os.path.basename(hd))
        for fpath in sorted(glob.glob(f"{hd}/*_input") + glob.glob(f"{hd}/power*_average")):
            base = os.path.basename(fpath)
            m = re.match(r"([a-z]+)(\d+)_(input|average)", base)
            if not m:
                continue
            kind, idx = m.group(1), m.group(2)
            label = read_text(f"{hd}/{kind}{idx}_label", f"{kind}{idx}")
            label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
            col = f"hwmon_{name}_{kind}_{label}" + ("_avg" if m.group(3) == "average" else "")
            col = re.sub(r"_+", "_", col)
            if col in info:
                col = f"{col}_{os.path.basename(hd)}"
            info[col] = {"temp": "mC", "power": "uW", "in": "mV", "curr": "mA",
                         "fan": "rpm", "energy": "uJ"}.get(kind, "")
            items.append((col, fpath))
    meta["hwmon_units"] = info
    return MultiFile(items)


def powercap_source(meta):
    items = []
    names = {}
    for zd in sorted(glob.glob("/sys/class/powercap/*")):
        if not os.path.exists(f"{zd}/energy_uj"):
            continue
        nm = read_text(f"{zd}/name", os.path.basename(zd))
        col = f"rapl_{os.path.basename(zd).replace(':', '_')}_{nm}_uj"
        names[col] = read_text(f"{zd}/max_energy_range_uj", "")
        items.append((col, f"{zd}/energy_uj"))
    meta["rapl_max_range_uj"] = names
    return MultiFile(items)


class AmdMsrEnergy:
    """AMD Fam17h+ per-core energy MSRs (root + msr module)."""
    UNIT, CORE, PKG = 0xC0010299, 0xC001029A, 0xC001029B
    only: set | None = None

    def __init__(self, meta):
        self.fds, self.cols = [], []
        vendor = ""
        for line in read_text("/proc/cpuinfo").splitlines():
            if line.startswith("vendor_id"):
                vendor = line.split(":", 1)[1].strip()
                break
        if vendor not in ("AuthenticAMD", "HygonGenuine"):
            return  # Intel per-core energy is not exposed; RAPL powercap covers the package
        cores = physical_cores(logical_cpus())
        try:
            fd0 = os.open("/dev/cpu/0/msr", os.O_RDONLY)
            unit = int.from_bytes(os.pread(fd0, 8, self.UNIT), "little")
            esu = (unit >> 8) & 0x1F
            meta["msr_energy_unit_J"] = 1.0 / (1 << esu)
            self.fds.append((fd0, self.PKG))
            self.cols.append("msr_pkg_energy_raw")
        except OSError as e:
            meta["msr_error"] = str(e)
            return
        for core in cores:
            cpu = core[0]
            if self.only is not None and cpu not in self.only:
                continue
            try:
                fd = os.open(f"/dev/cpu/{cpu}/msr", os.O_RDONLY)
                os.pread(fd, 8, self.CORE)
                self.fds.append((fd, self.CORE))
                self.cols.append(f"msr_core_energy_raw_cpu{cpu}")
            except OSError:
                pass

    def read(self):
        out = []
        for fd, reg in self.fds:
            try:
                out.append(int.from_bytes(os.pread(fd, 8, reg), "little") & 0xFFFFFFFF)
            except OSError:
                out.append(None)
        return out


def battery_source():
    items = []
    for ps in sorted(glob.glob("/sys/class/power_supply/*")):
        nm = os.path.basename(ps)
        for f in ("power_now", "current_now", "voltage_now", "energy_now", "charge_now",
                  "online", "capacity"):
            if os.path.exists(f"{ps}/{f}"):
                items.append((f"psu_{nm}_{f}", f"{ps}/{f}"))
    return MultiFile(items)


class KVSource:
    def __init__(self, path, keys, prefix, sep=None, value_idx=1):
        self.f = PFile(path)
        self.keys, self.prefix, self.sep, self.vi = keys, prefix, sep, value_idx
        d = self._parse()
        self.keys = [k for k in keys if k in d]
        self.cols = [f"{prefix}{k}" for k in self.keys]

    def _parse(self):
        d = {}
        for line in self.f.read().splitlines():
            p = line.replace(":", " ").split() if self.sep is None else line.split(self.sep)
            if len(p) > self.vi:
                try:
                    d[p[0]] = int(p[self.vi])
                except ValueError:
                    pass
        return d

    def read(self):
        d = self._parse()
        return [d.get(k) for k in self.keys]


class PSISource:
    def __init__(self):
        self.files, self.cols = [], []
        for res in ("cpu", "memory", "io", "irq"):
            f = try_pfile(f"/proc/pressure/{res}")
            if f is None:
                continue
            for kind in ("some", "full"):
                if kind in f.read():
                    self.files.append((f, kind))
                    self.cols.append(f"psi_{res}_{kind}_total_us")

    def read(self):
        out = []
        cache = {}
        for f, kind in self.files:
            txt = cache.setdefault(f.path, f.read())
            m = re.search(rf"^{kind} .*total=(\d+)", txt, re.M)
            out.append(int(m.group(1)) if m else None)
        return out


class NetDev:
    def __init__(self, iface="lo"):
        self.f = PFile("/proc/net/dev")
        self.iface = iface
        self.cols = [f"net_{iface}_{x}" for x in ("rx_bytes", "rx_packets", "tx_bytes", "tx_packets")]

    def read(self):
        for line in self.f.read().splitlines():
            if line.strip().startswith(self.iface + ":"):
                p = line.split(":", 1)[1].split()
                return [int(p[0]), int(p[1]), int(p[8]), int(p[9])]
        return [None] * 4


class TcpSnmp:
    KEYS = ["ActiveOpens", "PassiveOpens", "CurrEstab", "InSegs", "OutSegs", "RetransSegs"]

    def __init__(self):
        self.f = PFile("/proc/net/snmp")
        self.cols = [f"tcp_{k}" for k in self.KEYS]

    def read(self):
        lines = [l for l in self.f.read().splitlines() if l.startswith("Tcp:")]
        if len(lines) < 2:
            return [None] * len(self.KEYS)
        h, v = lines[0].split()[1:], lines[1].split()[1:]
        d = dict(zip(h, v))
        return [int(d[k]) if k in d else None for k in self.KEYS]


class LoadAvg:
    def __init__(self):
        self.f = PFile("/proc/loadavg")
        self.cols = ["loadavg_1m_x100"]

    def read(self):
        return [int(float(self.f.read().split()[0]) * 100)]


MEMINFO_KEYS = ["MemFree", "MemAvailable", "Buffers", "Cached", "Dirty", "Writeback",
                "AnonPages", "Mapped", "Shmem", "Slab", "PageTables", "AnonHugePages"]
VMSTAT_KEYS = ["pgfault", "pgmajfault", "pgalloc_normal", "pgalloc_dma32", "pgfree",
               "pgactivate", "pgdeactivate", "pgpgin", "pgpgout", "pswpin", "pswpout",
               "numa_hit", "numa_miss", "numa_local", "thp_fault_alloc", "compact_stall",
               "pgscan_kswapd", "pgsteal_kswapd", "nr_dirtied", "nr_written",
               "workingset_refault_anon", "workingset_refault_file", "nr_tlb_remote_flush",
               "nr_tlb_remote_flush_received", "nr_tlb_local_flush_all", "nr_tlb_local_flush_one"]


# --------------------------------------------------------------------------- #
# perf stat -I (system-wide, per CPU) for named PMU events
# --------------------------------------------------------------------------- #
PERF_CLI_CANDIDATES = [
    # Vendor-neutral. Each is probed at start-up; unsupported ones are dropped.
    # x86 free-running MSR PMU: true average frequency (APERF/MPERF) at no PMC cost
    "msr/aperf/", "msr/mperf/", "msr/tsc/",
    # RAPL energy (Intel: pkg/cores/ram/gpu/psys; AMD: pkg)
    "power/energy-pkg/", "power/energy-cores/", "power/energy-ram/",
    "power/energy-gpu/", "power/energy-psys/",
    # core events system-wide, *including* kernel and other tasks on the CPU
    "cycles", "instructions", "cache-misses", "branch-misses",
]
# Vendor-specific extras go in via --perf-cli-events, e.g. AMD Zen 2:
#   fp_ret_sse_avx_ops.all,l2_cache_req_stat.ic_dc_miss_in_l2,l3_comb_clstr_state.request_miss
# or Intel:  fp_arith_inst_retired.scalar_single,mem_load_retired.l3_miss


def perf_cli_available_events(candidates: list[str], perf: str) -> list[str]:
    ok = []
    for ev in candidates:
        try:
            r = subprocess.run([perf, "stat", "-a", "-x", ",", "-e", ev, "--", "sleep", "0.01"],
                               capture_output=True, text=True, timeout=10)
        except Exception:
            continue
        txt = r.stderr + r.stdout
        if r.returncode == 0 and "not supported" not in txt and "<not counted>" not in txt \
                and "event syntax error" not in txt:
            ok.append(ev)
    return ok


def start_perf_cli(out_path: str, events: list[str], interval_ms: int, cpu: int, meta: dict):
    perf = shutil.which("perf")
    if not perf:
        meta["perf_cli"] = "perf not found"
        return None
    evs = perf_cli_available_events(events, perf)
    meta["perf_cli_events"] = evs
    meta["perf_cli_rejected"] = [e for e in events if e not in evs]
    if not evs:
        meta["perf_cli"] = "no usable events (need perf_event_paranoid<=0 or root)"
        return None
    cmd = ["taskset", "-c", str(cpu), perf, "stat", "-A", "-x", ",",
           *(["-C", meta["perf_cli_cpus"]] if meta.get("perf_cli_cpus") else ["-a"]),
           "-I", str(interval_ms), "-o", out_path, "-e", ",".join(evs)]
    meta["perf_cli_cmd"] = " ".join(cmd)
    meta["perf_cli_start_mono_ns"] = mono_ns()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         preexec_fn=os.setsid)
    meta["perf_cli_start_mono_ns_after"] = mono_ns()
    return p


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cpu", type=int, default=-1, help="pin the sampler here")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--perf-cli", choices=["auto", "off"], default="auto")
    ap.add_argument("--perf-cli-interval-ms", type=int, default=100)
    ap.add_argument("--perf-cli-events", default="", help="comma list overriding defaults")
    ap.add_argument("--no-msr", action="store_true")
    ap.add_argument("--cpus", default="", help="only log per-CPU columns for these (e.g. 0,1,2)")
    a = ap.parse_args(argv)

    pin_self(a.cpu)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    os.makedirs(a.out, exist_ok=True)

    cpus = [c.cpu for c in logical_cpus()] or list(range(os.cpu_count() or 1))
    # include every online cpu, even ones outside our affinity mask
    try:
        from hwsig.topology import _parse_list  # type: ignore
    except ImportError:
        from .topology import _parse_list
    cpus = list(_parse_list(read_text("/sys/devices/system/cpu/online", "0"))) or cpus
    if a.cpus:
        want = {int(x) for x in a.cpus.split(",") if x != ""}
        cpus = [c for c in cpus if c in want]
        AmdMsrEnergy.only = set(want)

    meta: dict = {"cpus": cpus, "hz": a.hz}
    sources = []

    def add(name, ctor):
        try:
            s = ctor()
            if s.cols:
                sources.append(s)
                meta.setdefault("sources", []).append(name)
        except Exception as e:  # noqa: BLE001
            meta.setdefault("source_errors", {})[name] = repr(e)

    add("proc_stat", lambda: ProcStat(cpus))
    add("cpufreq", lambda: cpufreq_source(cpus))
    add("cpuidle", lambda: cpuidle_source(cpus, meta))
    add("interrupts", lambda: TableByCPU("/proc/interrupts", cpus, "irq", True))
    add("softirqs", lambda: TableByCPU("/proc/softirqs", cpus, "softirq", False))
    add("schedstat", lambda: SchedStat(cpus))
    add("hwmon", lambda: hwmon_source(meta))
    add("powercap", lambda: powercap_source(meta))
    if not a.no_msr:
        add("amd_msr_energy", lambda: AmdMsrEnergy(meta))
    add("power_supply", battery_source)
    add("meminfo", lambda: KVSource("/proc/meminfo", MEMINFO_KEYS, "mem_"))
    add("vmstat", lambda: KVSource("/proc/vmstat", VMSTAT_KEYS, "vm_"))
    add("psi", PSISource)
    add("net_lo", NetDev)
    add("tcp", TcpSnmp)
    add("loadavg", LoadAvg)

    perf_proc = None
    if a.perf_cli == "auto":
        evs = [e for e in a.perf_cli_events.split(",") if e] or PERF_CLI_CANDIDATES
        if a.cpus:
            meta["perf_cli_cpus"] = a.cpus
        perf_proc = start_perf_cli(os.path.join(a.out, "perf_cli.csv"), evs,
                                   a.perf_cli_interval_ms, a.cpu, meta)

    cols = ["t_mono_ns"] + [c for s in sources for c in s.cols]
    meta["n_columns"] = len(cols)
    with open(os.path.join(a.out, "monitor_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    period = 1.0 / a.hz
    with open(os.path.join(a.out, "monitor.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        nxt = time.monotonic()
        n = 0
        while not _STOP:
            t0 = mono_ns()
            row = [t0]
            for s in sources:
                try:
                    row.extend(s.read())
                except Exception:  # noqa: BLE001
                    row.extend([None] * len(s.cols))
            w.writerow(row)
            n += 1
            if n % 50 == 0:
                f.flush()
            nxt += period
            dt = nxt - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.monotonic()

    if perf_proc is not None:
        try:
            os.killpg(perf_proc.pid, signal.SIGINT)
            perf_proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            perf_proc.kill()


if __name__ == "__main__":
    main()
