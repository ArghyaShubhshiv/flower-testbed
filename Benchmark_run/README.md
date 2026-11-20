# hwsig: hardware signature of an ALIE attack in Flower

Works on any x86 Linux machine. 6 Flower clients (CIFAR-10), each pinned to
its own core; the ALIE colluder runs in its own process on its own core.
Honest vs attack runs, then a search for any metric (or combination) that
predicts the attack. Metrics: see `METRICS.md`.

Run every command from the repository root (the folder that contains `hwsig/`).

## 1. Install (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r hwsig/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Optional, for extra metrics (frequency, energy and system-wide counters via `perf`):

```bash
sudo pacman -S perf                         # Arch
sudo apt install linux-tools-$(uname -r)    # Debian/Ubuntu
sudo dnf install perf                       # Fedora
```

## 2. Download CIFAR-10 and split it into 6 shards (once)

```bash
python -m hwsig.data --n-clients 6
```

## 3. Prepare the machine (each boot or session; optional but strongly recommended)

```bash
sudo bash hwsig/prep_host.sh       # counter access, fixed frequency, turbo off, IRQs moved off the test cores
python -m hwsig.check              # shows the CPU plan and which metrics are available
```

Plug laptops into AC power and close other heavy programs.

## 4. Pilot run (~5 min): check that the attack actually lowers accuracy

```bash
python -m hwsig.campaign --name pilot --pairs 1 --rounds 5 --warmup-runs 0
python -m hwsig.analyze results/pilot
cat results/pilot/analysis/runs.csv      # compare final_acc: honest vs attack
```

If the attack run's accuracy is not clearly lower, add `--z 2` (stronger
attack) to the campaign command below.

## 5. Full campaign (10 honest + 10 ALIE runs, ~2-3 h)

```bash
sudo .venv/bin/python -m hwsig.campaign --name alie_v1 --pairs 10 --rounds 20
```

`sudo` is only needed for AMD per-core energy registers; without it the run
still works with those columns missing. If the campaign is interrupted, rerun
the same command: finished runs are skipped.

## 6. Analyse

```bash
python -m hwsig.analyze results/alie_v1 --perm 100
xdg-open results/alie_v1/analysis/report.md
```

## 7. Restore the machine

```bash
sudo bash hwsig/prep_host.sh --restore
```

## Options worth knowing

| Flag (campaign) | Default | Meaning |
|---|---|---|
| `--n-clients` / `--n-byz` | 6 / 2 | number of clients / colluders (needs n_clients + 2 cores for clean pinning) |
| `--z` | 1.0 | ALIE strength: submitted = mean − z·std of the colluders' updates |
| `--no-shadow` | off | in honest runs, don't route colluders through the attacker process |
| `--agg` | fedavg | `median` / `trimmed_mean` |
| `--model` | small | `medium` = ~20x more compute per round |
| `--max-samples` | 0 (all) | cap samples per client for faster rounds |
| `--perf-cli-events` | – | extra vendor events, e.g. `fp_ret_sse_avx_ops.all` (AMD) |
| `--cpu-plan` | auto | `server=0,clients=1:2:3:4:5:6,attacker=7,housekeeping=8` |

## Output

```
results/<name>/campaign.json             config, CPU plan, machine description (once)
results/<name>/runs/<run>/
    meta.json                            mode, seed, timestamps, temperatures
    client_<i>.jsonl server.jsonl attacker.jsonl   per-round values that change
    *.const.json                         values constant within this run (stored once)
    monitor.csv                          20 Hz samples, only changing columns, only role CPUs
    perf_cli.csv                         (if perf is installed)
    logs/                                console output, kept only if the run failed
results/<name>/analysis/
    report.md  classifiers.csv  univariate_*.csv  rounds.csv  client_rounds.csv  *.png
```
