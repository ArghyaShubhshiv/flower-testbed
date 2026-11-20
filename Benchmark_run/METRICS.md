# Metric catalogue: what a Linux machine can tell you (Intel or AMD)

Every metric observable from user space or root on a typical x86 Linux machine,
the command to read it by hand, its scope, the privileges it needs, and the
column prefix it lands under in the harness. ✅ = collected automatically when
the machine exposes it (absent sources are skipped, never faked), ➕ = optional
extension (command given), ✖ = not available on that vendor.

**Logging policy:** per-round logs keep only values that change within a run.
Anything constant for a whole run is written once to `*.const.json` /
`monitor_meta.json["constants"]` (so it can still differ *between* runs), and
per-CPU columns are logged only for CPUs that host a role.

Scopes: **T** = per thread/process, **C** = per logical CPU (core),
**X** = per CCX / L3 slice, **P** = package/socket, **S** = system-wide.

Column prefixes in `analysis/*.csv`:
`<seg>.thr.*` perf, this thread · `<seg>.core.*` perf, the whole CPU ·
`<seg>.os.*` in-process OS counters · `<seg>.mon.*` monitor, this CPU over the window ·
`sys.*` monitor, system-wide · `srvcore.*` / `attcore.*` server/attacker CPU ·
`colluders.*` / `others.*` mean over colluder / honest clients ·
`model.*` update statistics (not hardware). `<seg>` ∈ idle, train, post, fit
(client), wait, agg, eval (server), idle, gather, compute, reply (attacker).

---

## 1. Core PMU: generic perf events (perf_event_open)

Cores typically have 4-8 programmable counters per logical CPU (Zen 2: 6). With
more events than that the kernel time-multiplexes them; the harness
scales by enabled/running and records the worst ratio as `mux_min` (only when
multiplexing happened).

| Metric | Manual command | Scope | Priv | Harness |
|---|---|---|---|---|
| cycles | `perf stat -e cycles -p PID` / `-C N` | T, C | paranoid ≤2 (T, user only) / ≤0 (C) | ✅ `thr.cycles`, `core.cycles` |
| instructions retired | `perf stat -e instructions` | T, C | same | ✅ |
| IPC (derived) | instructions / cycles | T, C | – | ✅ `.ipc` |
| cache-references / cache-misses | `perf stat -e cache-references,cache-misses` (on AMD these map to L2/L3 request events) | T, C | same | ✅ `.cache_miss_rate` |
| branch-instructions / branch-misses | `perf stat -e branches,branch-misses` | T, C | same | ✅ `.branch_miss_rate` |
| stalled-cycles-frontend / backend | `perf stat -e stalled-cycles-frontend,stalled-cycles-backend` | T, C | same | ✅ `.stall_frac_*` |
| L1D loads / misses / prefetches | `perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-prefetches` | T, C | same | ✅ `.l1d_miss_rate` |
| L1I loads / misses | `perf stat -e L1-icache-loads,L1-icache-load-misses` | T, C | same | ✅ |
| LLC loads / misses | `perf stat -e LLC-loads,LLC-load-misses` | T, C | same | ✅ (often unsupported on Zen 2: use the L3 uncore events in §3) |
| dTLB / iTLB loads / misses | `perf stat -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses` | T, C | same | ✅ |
| BPU loads / misses | `perf stat -e branch-loads,branch-load-misses` | T, C | same | ✅ |
| effective GHz (derived) | cycles / task-clock | T, C | – | ✅ `.ghz_eff` |
| ref-cycles | `perf stat -e ref-cycles` | C | – | ➕ Intel only (TSC-rate cycles) |

## 2. Kernel software events (perf_event_open, PERF_TYPE_SOFTWARE)

| Metric | Command | Scope | Harness |
|---|---|---|---|
| task-clock | `perf stat -e task-clock` | T, C | ✅ `thr.task_clock_ns` |
| context-switches | `perf stat -e cs` | T, C | ✅ (0 at paranoid 2, because these are kernel-side) |
| cpu-migrations | `perf stat -e migrations` | T, C | ✅ (should be 0: pinning check) |
| minor / major faults | `perf stat -e minor-faults,major-faults` | T, C | ✅ |

## 3. Named / vendor PMU events (via `perf stat -A -I 100`, monitor)

Collected by default (vendor-neutral, each probed at start-up): `msr/aperf/`,
`msr/mperf/`, `msr/tsc/`, `power/energy-*/`, and system-wide `cycles`,
`instructions`, `cache-misses`, `branch-misses`. Vendor-specific events are added with
`--perf-cli-events a,b,c`; names come from `perf list` and vary by kernel and CPU. The
AMD Zen 2 rows below are examples (Intel equivalents: `fp_arith_inst_retired.*`,
`mem_load_retired.l3_miss`, `cycle_activity.stalls_*`, `uncore_imc/cas_count_read/`).

| Metric | Event | Scope | Harness |
|---|---|---|---|
| retired branch mispredicts (AMD) | `ex_ret_brn_misp` | C | ➕ |
| retired SSE/AVX FLOPs (AMD) | `fp_ret_sse_avx_ops.all` | C | ➕ (training is FP-bound; ALIE arithmetic is FP too) |
| L1 DTLB misses (AMD) | `ls_l1_d_tlb_miss.all` | C | ➕ |
| L2 hits / misses (AMD) | `l2_cache_req_stat.ic_dc_hit_in_l2`, `…ic_dc_miss_in_l2` | C | ➕ |
| demand fills from DRAM (AMD) | `ls_refills_from_sys.ls_mabresp_lcl_dram` | C | ➕ |
| instruction-fetch stalls (AMD) | `ic_fetch_stall.ic_stall_any` | C | ➕ |
| L3 accesses / misses (AMD uncore, per CCX) | `l3_request_g1.caching_l3_cache_accesses`, `l3_comb_clstr_state.request_miss` | X | ➕ |
| APERF / MPERF / TSC (free-running) | `msr/aperf/`, `msr/mperf/`, `msr/tsc/` | C | ✅ `.aperf_mperf_ratio`: the true average frequency in C0, with no PMC cost |
| IRPERF (AMD, retired instr, free-running) | `msr/irperf/` | C | ➕ |
| dispatch stalls by resource | `de_dis_dispatch_token_stalls*` | C | ➕ add with `--perf-cli-events` |
| DRAM bandwidth | AMD `amd_df/…/`, Intel `uncore_imc/cas_count_read/` | P | ➕ |
| IBS sampling (AMD precise per-instruction latency) | `perf record -e ibs_op// -C N` | C | ➕ |
| LBR / PEBS (Intel precise sampling) | `perf record -e cycles:pp --call-graph lbr` | C | ➕ Intel only |

## 4. Energy and power

| Metric | Command | Scope | Priv | Harness |
|---|---|---|---|---|
| RAPL energy: package; Intel also core (PP0), uncore/GPU (PP1), DRAM, psys | `cat /sys/class/powercap/intel-rapl:*/energy_uj` (name in `…/name`) | P / domain | root since CVE-2020-8694 (`prep_host.sh` chmods it) | ✅ `sys.rapl_*_J`, `_W` |
| package energy (perf) | `perf stat -a -e power/energy-pkg/` | P | paranoid ≤0 | ✅ `sys.perf_power_energy_pkg` |
| **per-core energy, AMD only** (MSR CORE_ENERGY_STAT 0xC001029A) | `sudo rdmsr -p N 0xc001029a` (msr-tools), unit from 0xC0010299 bits 12:8 | C (physical core) | root + `modprobe msr` | ✅ `mon.msr_core_energy_J/_W` (campaign run as root) |
| package energy MSR, AMD (0xC001029B) | `sudo rdmsr -p 0 0xc001029b` | P | root | ✅ `sys.msr_pkg_energy_J/_W` |
| per-core power, C-states, Bzy_MHz in one view | `sudo turbostat --interval 1 --show Core,CPU,Busy%,Bzy_MHz,CorWatt,PkgWatt,C1%,C2%` | C | root | ➕ cross-check |
| battery discharge power | `cat /sys/class/power_supply/BAT*/power_now` (µW) | S | – | ✅ `sys.psu_*` (only meaningful on battery: run on AC instead) |
| core voltage / current | AMD: `zenpower3` module (SVI2); Intel: `sudo turbostat` (CoreV) | P | module / root | ➕ appears under `hwmon_*` automatically if a driver exposes it |
| firmware power telemetry | AMD `ryzenadj --info`; Intel `turbostat` | P | root | ➕ |
| GPU power | `cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_average` (amdgpu); `nvidia-smi --query-gpu=power.draw --format=csv` | GPU | – | ✅ amdgpu via hwmon, ➕ NVIDIA |
| wall power | external meter (e.g. a USB-PD or mains logger) | system | – | ➕ the only number not filtered by firmware |

## 5. Frequency, idle states, thermal

| Metric | Command | Scope | Harness |
|---|---|---|---|
| current frequency | `cat /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq` | C | ✅ `mon.freq_khz` (+`__max`) |
| actual frequency in C0 | APERF/MPERF (§3), `cpuinfo_cur_freq` (root) | C | ✅ |
| C-state entries and residency | `cat /sys/devices/system/cpu/cpuN/cpuidle/state*/{name,usage,time}` | C | ✅ `mon.cstateK_usage`, `_time_us` |
| CPU temperature (AMD k10temp Tctl/Tccd; Intel coretemp per-core + package) | `cat /sys/class/hwmon/hwmon*/temp*_input`, `sensors` | P / core | ✅ `sys.hwmon_<driver>_temp_*` |
| GPU, NVMe, ACPI zone temps; fan RPM | `sensors` | – | ✅ `sys.hwmon_*` |
| thermal throttling events | `/proc/interrupts` THR row; Intel `/sys/devices/system/cpu/cpuN/thermal_throttle/*` | C | ✅ `mon.irq_THR` |

## 6. Scheduler and CPU time

| Metric | Command | Scope | Harness |
|---|---|---|---|
| user/nice/system/idle/iowait/irq/softirq/steal jiffies | `cat /proc/stat` (`cpuN` lines), `mpstat -P ALL 1` | C | ✅ `mon.stat_*`, `mon.busy_frac` |
| run time / runqueue wait / timeslices | `cat /proc/schedstat` (fields 7-9 per CPU) | C | ✅ `mon.sched_*` |
| process run/wait/slices | `cat /proc/PID/schedstat` | T | ✅ `os.rq_run_ns`, `os.rq_wait_ns`, `os.timeslices` |
| per-thread migrations, wait_sum, wakeups | `cat /proc/PID/task/TID/sched` | T | ✅ `os.migrations`, `os.wait_sum_ms`… (needs `kernel.sched_schedstats=1`, set by prep_host) |
| voluntary / involuntary ctx switches | `getrusage`; `pidstat -w` | T | ✅ `os.thr_vol_cs`, `os.proc_invol_cs`… |
| system ctx switches, forks, running/blocked procs | `/proc/stat` ctxt, processes, procs_running | S | ✅ `sys.sys_*` |
| CPU pressure (PSI) | `cat /proc/pressure/cpu` | S | ✅ `sys.psi_cpu_*` |
| load average | `cat /proc/loadavg` | S | ✅ |
| thread / process CPU time, user vs system | `time.thread_time_ns()`, `getrusage(RUSAGE_THREAD)` | T | ✅ `os.thread_cpu_ns`, `os.thr_utime_s`, `os.thr_stime_s` |
| wall time per segment | CLOCK_MONOTONIC | T | ✅ `os.wall_ns`, `fit.wall_ns`, `round.client_fit_spread_ns` |

## 7. Interrupts and IPIs

| Metric | Command | Scope | Harness |
|---|---|---|---|
| device IRQs (sum) | `cat /proc/interrupts` (numeric rows) | C | ✅ `mon.irq_dev_total` |
| local timer (LOC), rescheduling IPI (RES), function-call IPI (CAL), TLB shootdown (TLB), NMI, PMI, IRQ-work (IWI), spurious, thermal (THR), deferred error (DFR) | `/proc/interrupts` named rows | C | ✅ `mon.irq_<ROW>` |
| softirqs: TIMER, NET_RX/TX, SCHED, RCU, TASKLET, HRTIMER, BLOCK | `cat /proc/softirqs` | C | ✅ `mon.softirq_*` |
| IRQ pressure | `cat /proc/pressure/irq` | S | ✅ if kernel has it |

## 8. Memory

| Metric | Command | Scope | Harness |
|---|---|---|---|
| RSS, peak RSS, thread count | `cat /proc/PID/status` | T | ✅ `os.rss_kB`, `os.hwm_kB`, `os.threads` |
| minor/major faults, block I/O ops | `getrusage` | T | ✅ `os.thr_minflt`, `os.proc_blk_in`… |
| bytes / syscalls read and written | `cat /proc/PID/io` | T | ✅ `os.io_rchar/wchar/syscr/syscw` (gRPC traffic) |
| system meminfo | `cat /proc/meminfo` | S | ✅ `sys.mem_*` |
| page alloc/free, faults, TLB flush counts, THP, compaction | `cat /proc/vmstat` | S | ✅ `sys.vm_*` |
| memory / IO pressure | `cat /proc/pressure/{memory,io}` | S | ✅ `sys.psi_*` |
| Python GC collections per generation | `gc.get_stats()` | T | ✅ `os.gc0..2` |

## 9. Network (all traffic is gRPC over loopback)

| Metric | Command | Scope | Harness |
|---|---|---|---|
| lo bytes / packets | `cat /proc/net/dev` | S | ✅ `sys.net_lo_*` |
| TCP segments in/out, retransmits, opens | `cat /proc/net/snmp` | S | ✅ `sys.tcp_*` |
| per-socket stats | `ss -tin` | socket | ➕ |
| payload sizes | Flower message size; attacker `bytes_in` | T | ✅ `attacker.bytes_in` |

## 10. Application / ML level (kept separate as `model.*`)

| Metric | Where | Harness |
|---|---|---|
| train loss / accuracy per round | client | ✅ `model.train_loss`, `model.train_acc` |
| own update L2 norm, submitted update norm, cosine(own, submitted) | client | ✅ `model.*`: the classic norm/cosine defence baseline |
| global test accuracy / loss | server | ✅ `runs.csv`, `accuracy.png` |
| aggregation / evaluation time | server | ✅ `server.agg.*`, `server.eval.*` |

## 11. Things that are *not* observable here (and why)

* **Per-core energy on Intel**: RAPL is per package/domain only.
* **Per-core temperature on AMD**: k10temp exposes Tctl and per-CCD Tccd only.
* **Per-core voltage**: rail-level at best.
* **Inside a VM / container**: PMU events, RAPL and MSRs are usually hidden or
  virtualised. Run on bare metal (the harness still runs in a VM, with those
  metrics missing).
* **EM / acoustic / power-line side channels**: need external instruments.
