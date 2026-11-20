"""Launch one pinned experiment: monitor, colluder, server, N clients.

Each process is pinned *before exec* (preexec_fn -> sched_setaffinity), so even
interpreter start-up and imports stay on the assigned CPU.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .common import compact_run, mono_ns, read_text


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_port(port: int, timeout: float) -> bool:
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _uniq(paths: list[str]) -> str:
    vals = sorted({read_text(p) for p in paths if os.path.exists(p)} - {""})
    return ",".join(vals)


def host_snapshot() -> dict:
    """Machine description, recorded ONCE per campaign (campaign.json)."""
    cpu = Path("/sys/devices/system/cpu")
    model = next((l.split(":", 1)[1].strip() for l in read_text("/proc/cpuinfo").splitlines()
                  if l.startswith("model name")), "")
    snap = {
        "kernel": platform.release(),
        "cpu_model": model,
        "logical_cpus": os.cpu_count(),
        "scaling_driver": read_text(str(cpu / "cpu0/cpufreq/scaling_driver")),
        "governor": _uniq([str(p) for p in cpu.glob("cpu*/cpufreq/scaling_governor")]),
        "epp": _uniq([str(p) for p in cpu.glob("cpu*/cpufreq/energy_performance_preference")]),
        "boost": read_text(str(cpu / "cpufreq/boost")) or
                 ("off" if read_text(str(cpu / "intel_pstate/no_turbo")) == "1" else
                  "on" if (cpu / "intel_pstate/no_turbo").exists() else ""),
        "smt": read_text(str(cpu / "smt/control")),
        "isolated": read_text(str(cpu / "isolated")),
        "perf_event_paranoid": read_text("/proc/sys/kernel/perf_event_paranoid"),
        "irqbalance_running": subprocess.run(["pgrep", "-x", "irqbalance"],
                                             capture_output=True).returncode == 0,
        "on_ac": _uniq([str(p) for p in Path("/sys/class/power_supply").glob("*/online")]),
        "root": os.geteuid() == 0,
        "python": platform.python_version(),
    }
    try:
        import flwr, torch  # noqa: F401
        snap["flwr"], snap["torch"] = flwr.__version__, torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in snap.items() if v not in ("", None)}


def max_temp_c() -> float | None:
    """Hottest k10temp/coretemp/zenpower reading, in deg C."""
    best = None
    for hd in Path("/sys/class/hwmon").glob("hwmon*"):
        name = read_text(str(hd / "name"))
        if name not in ("k10temp", "zenpower", "coretemp", "cpu_thermal"):
            continue
        for t in hd.glob("temp*_input"):
            try:
                v = int(read_text(str(t))) / 1000.0
                best = v if best is None else max(best, v)
            except ValueError:
                pass
    return best


def _popen(cmd, cpu, log_path, cwd, env):
    lf = open(log_path, "w")
    pre = (lambda: os.sched_setaffinity(0, {cpu})) if cpu is not None and cpu >= 0 else None
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                            preexec_fn=pre, start_new_session=True), lf


def run(cfg: dict, plan: dict, out_dir: Path, repo: Path) -> dict:
    """cfg keys: run_id, mode('honest'|'attack'), seed, rounds, n_clients, n_byz,
    shadow(bool), z, agg, data, max_samples, model, epochs, batch, lr,
    monitor_hz, perf_cli(bool), pre_s, post_s, timeout_s, test_samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    py = sys.executable
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    port = _free_port()
    sock = f"/tmp/hwsig_{os.getpid()}_{cfg['run_id']}.sock"
    byz = set(plan["byz_ids"])
    use_attacker = cfg["n_byz"] > 0 and (cfg["mode"] == "attack" or cfg["shadow"])

    meta = {"cfg": {k: cfg[k] for k in ("run_id", "mode", "seed", "pair") if k in cfg},
            "temp_pre_c": max_temp_c(),
            "colluder_ids": sorted(byz) if use_attacker else [],
            "byzantine_ids": sorted(byz) if cfg["mode"] == "attack" else [],
            "t": {}}
    procs: dict[str, tuple[subprocess.Popen, object]] = {}

    def mark(k):
        meta["t"][k] = mono_ns()

    def stop(name, sig=signal.SIGTERM, wait=10):
        p, lf = procs.get(name, (None, None))
        if p is None:
            return
        if p.poll() is None:
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        lf.close()

    ok = True
    try:
        mark("start")
        mon_cmd = [py, "-m", "hwsig.monitor", "--out", str(out_dir), "--cpu", str(plan["housekeeping"]),
                   "--hz", str(cfg["monitor_hz"]), "--perf-cli", "auto" if cfg["perf_cli"] else "off",
                   "--cpus", ",".join(map(str, plan["role_cpus"]))]
        if cfg.get("perf_cli_events"):
            mon_cmd += ["--perf-cli-events", cfg["perf_cli_events"]]
        procs["monitor"] = _popen(mon_cmd, plan["housekeeping"], out_dir / "logs/monitor.log", repo, env)
        time.sleep(cfg["pre_s"])
        mark("baseline_end")

        if use_attacker:
            procs["attacker"] = _popen(
                [py, "-m", "hwsig.attacker", "--addr", sock, "--n-colluders", str(len(byz)),
                 "--mode", "attack" if cfg["mode"] == "attack" else "shadow",
                 "--z", str(cfg["z"]), "--cpu", str(plan["attacker"]),
                 "--log", str(out_dir / "attacker.jsonl"), "--run-id", cfg["run_id"]],
                plan["attacker"], out_dir / "logs/attacker.log", repo, env)

        procs["server"] = _popen(
            [py, "-m", "hwsig.fl_server", "--addr", f"127.0.0.1:{port}", "--cpu", str(plan["server"]),
             "--rounds", str(cfg["rounds"]), "--n-clients", str(cfg["n_clients"]),
             "--agg", cfg["agg"], "--data", cfg["data"], "--model", cfg["model"],
             "--test-samples", str(cfg["test_samples"]), "--seed", str(cfg["seed"]),
             "--log", str(out_dir / "server.jsonl"), "--run-id", cfg["run_id"]],
            plan["server"], out_dir / "logs/server.log", repo, env)
        if not _wait_port(port, 120):
            raise RuntimeError("server did not open its port")
        mark("server_up")

        for cid, cpu in enumerate(plan["clients"]):
            cmd = [py, "-m", "hwsig.fl_client", "--cid", str(cid), "--cpu", str(cpu),
                   "--server", f"127.0.0.1:{port}", "--data", cfg["data"],
                   "--n-clients", str(cfg["n_clients"]), "--max-samples", str(cfg["max_samples"]),
                   "--model", cfg["model"], "--epochs", str(cfg["epochs"]),
                   "--batch", str(cfg["batch"]), "--lr", str(cfg["lr"]), "--seed", str(cfg["seed"]),
                   "--log", str(out_dir / f"client_{cid}.jsonl"), "--run-id", cfg["run_id"]]
            if use_attacker and cid in byz:
                cmd += ["--colluder", "--attacker-addr", sock]
            procs[f"client{cid}"] = _popen(cmd, cpu, out_dir / f"logs/client{cid}.log", repo, env)
        mark("clients_started")

        p_srv = procs["server"][0]
        t_end = time.time() + cfg["timeout_s"]
        while p_srv.poll() is None:
            if time.time() > t_end:
                raise TimeoutError("server timeout")
            for name, (p, _) in procs.items():
                if name.startswith("client") and p.poll() not in (None, 0):
                    raise RuntimeError(f"{name} exited with {p.returncode}")
            time.sleep(0.5)
        if p_srv.returncode != 0:
            raise RuntimeError(f"server exited with {p_srv.returncode}")
        mark("server_done")
        for cid in range(cfg["n_clients"]):
            try:
                procs[f"client{cid}"][0].wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        mark("clients_done")
    except Exception as e:  # noqa: BLE001
        ok = False
        meta["error"] = repr(e)
    finally:
        for name in [n for n in procs if n.startswith("client")] + ["server", "attacker"]:
            stop(name)
        time.sleep(cfg["post_s"] if ok else 0)
        mark("end")
        stop("monitor", signal.SIGTERM, 15)
        try:
            os.unlink(sock)
        except OSError:
            pass
        meta["temp_post_c"] = max_temp_c()
        meta["ok"] = ok
        try:
            compact_run(out_dir)
        except Exception as e:  # noqa: BLE001
            meta["compact_error"] = repr(e)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=1, default=str)
        if ok:
            shutil.rmtree(out_dir / "logs", ignore_errors=True)   # console output kept only on failure
            (out_dir / "DONE").write_text("ok\n")
    return meta
