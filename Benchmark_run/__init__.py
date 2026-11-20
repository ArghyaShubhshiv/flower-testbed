"""Hardware-signature testbed for Byzantine (ALIE) attacks in Flower FL.

Layout
------
common.py      timing, pinning, JSONL helpers
topology.py    CPU topology discovery and the role -> CPU plan
perfctr.py     perf_event_open(2) via ctypes (per-thread and per-core counters)
procstats.py   in-process OS/software counters (rusage, /proc/self/*)
monitor.py     system sampler process (procfs/sysfs/hwmon/RAPL/MSR + perf stat -I)
models.py      CIFAR-10 models
data.py        CIFAR-10 download, IID partition, shard loading
attacker.py    ALIE colluder process (own pinned core)
fl_server.py   Flower server (legacy gRPC) with instrumentation
fl_client.py   Flower client with instrumentation
run_one.py     launch one pinned experiment
campaign.py    n honest + n ALIE runs, interleaved and randomised
analyze.py     feature tables, statistics, classifiers, report
"""
