"""Pre-flight: what can this machine measure right now?

    python -m hwsig.check [--n-clients 6 --n-byz 2]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

from . import perfctr
from .common import read_text
from .topology import describe, make_plan


def line(ok: bool, text: str) -> None:
    print(f"  [{'x' if ok else ' '}] {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=6)
    ap.add_argument("--n-byz", type=int, default=2)
    a = ap.parse_args()

    print("CPU plan")
    print("  " + describe(make_plan(a.n_clients, a.n_byz)).replace("\n", "\n  "))

    pr = perfctr.probe()
    hw = [e for e in pr["thread_ok"] if e not in ("task_clock_ns", "context_switches",
                                                   "cpu_migrations", "minor_faults", "major_faults")]
    print("\nCounters")
    line(bool(hw), f"hardware PMU events, per thread: {len(hw)} "
                   f"({'user+kernel' if not pr['thread_exclude_kernel'] else 'user only'})")
    line(len(pr["core_ok"]) > 0, f"per-CPU events: {len(pr['core_ok'])}  (needs perf_event_paranoid <= 0)")
    print(f"      perf_event_paranoid = {pr['paranoid']}")

    print("\nEnergy / power")
    rapl = glob.glob("/sys/class/powercap/*/energy_uj")
    line(bool(rapl) and all(os.access(z, os.R_OK) for z in rapl),
         f"RAPL powercap zones readable: {len(rapl)} found")
    vendor = next((l.split(":", 1)[1].strip() for l in read_text("/proc/cpuinfo").splitlines()
                   if l.startswith("vendor_id")), "")
    if vendor in ("AuthenticAMD", "HygonGenuine"):
        ok = False
        try:
            fd = os.open("/dev/cpu/0/msr", os.O_RDONLY)
            os.pread(fd, 8, 0xC001029A)
            ok = True
        except OSError:
            pass
        line(ok, "AMD per-core energy MSR  (needs root + 'modprobe msr')")
    bats = glob.glob("/sys/class/power_supply/*/online")
    if bats:
        line(any(read_text(b) == "1" for b in bats), "on AC power")

    print("\nSensors (hwmon)")
    for hd in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        n = read_text(f"{hd}/name")
        k = len(glob.glob(f"{hd}/*_input"))
        print(f"      {n:14s} {k} inputs")

    print("\nFrequency policy")
    cpu = "/sys/devices/system/cpu"
    gov = read_text(f"{cpu}/cpu0/cpufreq/scaling_governor")
    line(gov == "performance", f"governor = {gov or 'n/a'}")
    boost = read_text(f"{cpu}/cpufreq/boost")
    nt = read_text(f"{cpu}/intel_pstate/no_turbo")
    off = boost == "0" or nt == "1"
    line(off, f"turbo/boost disabled  (boost={boost or 'n/a'}, intel no_turbo={nt or 'n/a'})")

    print("\nperf CLI (optional)")
    perf = shutil.which("perf")
    if not perf:
        line(False, "perf not installed -> APERF/MPERF, RAPL perf events, system-wide counters skipped")
    else:
        from .monitor import PERF_CLI_CANDIDATES, perf_cli_available_events
        ok = perf_cli_available_events(PERF_CLI_CANDIDATES, perf)
        line(bool(ok), f"usable events: {', '.join(ok) or 'none'}")

    print("\nEverything unchecked still runs; those metrics are simply absent.")


if __name__ == "__main__":
    main()
