"""ALIE colluder: a separate process on its own pinned core.

Threat model (Baruch et al., NeurIPS 2019, *non-omniscient* variant): the
adversary controls ``m`` clients and only sees *their* data. Each colluding
client trains honestly, ships its update to this process, and this process
returns the malicious update

    u_byz = mean(U_colluders) - z * std(U_colluders)        (coordinate-wise)

which every colluder then submits instead of its own. The paper uses z = 1 for
CIFAR-10 with 20 % attackers; z_max from the paper's formula is <= 0 for n = 6,
so we use a fixed z (``--z``, default 1.0).

``--mode shadow`` is the control: identical sockets, identical barrier, same
bytes moved, but the update comes back unchanged and no ALIE arithmetic runs.
Running the honest condition in shadow mode means the honest/attack contrast
isolates *the attack itself*, not "an extra process exists" or "two clients do
extra IPC".
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hwsig.common import single_thread_env, pin_self, mono_ns, JsonlWriter  # type: ignore
else:
    from .common import single_thread_env, pin_self, mono_ns, JsonlWriter

single_thread_env()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", required=True, help="unix socket path")
    ap.add_argument("--authkey", default="hwsig")
    ap.add_argument("--n-colluders", type=int, required=True)
    ap.add_argument("--mode", choices=["attack", "shadow"], required=True)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--cpu", type=int, default=-1)
    ap.add_argument("--log", required=True)
    ap.add_argument("--run-id", default="")
    a = ap.parse_args(argv)

    pin_self(a.cpu)
    import numpy as np
    from multiprocessing.connection import Listener, wait
    if __package__ in (None, ""):
        from hwsig import perfctr, procstats  # type: ignore
    else:
        from . import perfctr, procstats

    log = JsonlWriter(a.log)
    th = perfctr.CounterSet("thread")
    core = perfctr.CounterSet("core", cpu=a.cpu) if a.cpu >= 0 else None
    log.write({"event": "start", "mode": a.mode, "z": a.z})

    def snap():
        return (procstats.snapshot(), th.read(), core.read() if core else {}, mono_ns())

    def seg(name, s0, s1):
        d = procstats.delta(s0[0], s1[0], f"{name}.os.")
        d.update(perfctr.delta(s0[1], s1[1], f"{name}.thr."))
        if core:
            d.update(perfctr.delta(s0[2], s1[2], f"{name}.core."))
        return d

    if os.path.exists(a.addr):
        os.unlink(a.addr)
    listener = Listener(a.addr, family="AF_UNIX", authkey=a.authkey.encode())
    signal.signal(signal.SIGTERM, lambda *_: (listener.close(), sys.exit(0)))

    conns = {}
    while len(conns) < a.n_colluders:
        c = listener.accept()
        hello = c.recv()
        conns[hello["cid"]] = c
    log.write({"event": "all_connected", "cids": sorted(conns), "t": mono_ns()})

    s_prev = snap()
    while conns:
        pending: dict[int, dict] = {}
        bufs: dict[int, bytes] = {}
        s_first = None
        # ---- gather one update from every colluder (barrier) ----
        while len(pending) < len(conns):
            ready = wait([c for cid, c in conns.items() if cid not in pending])
            for c in ready:
                cid = next(k for k, v in conns.items() if v is c)
                try:
                    hdr = c.recv()
                    buf = c.recv_bytes()
                except (EOFError, OSError):
                    conns.pop(cid, None)
                    continue
                if s_first is None:
                    s_first = snap()
                pending[cid] = hdr
                bufs[cid] = buf
            if not conns:
                break
        if not conns or not pending:
            break
        s_gathered = snap()
        rnd = next(iter(pending.values()))["round"]

        # ---- the attack (or nothing, in shadow mode) ----
        cids = sorted(pending)
        if a.mode == "attack":
            H = np.stack([np.frombuffer(bufs[c], dtype=np.float32) for c in cids]).astype(np.float64)
            mu = H.mean(axis=0)
            sigma = H.std(axis=0)
            mal = (mu - a.z * sigma).astype(np.float32)
            out = {c: mal.tobytes() for c in cids}
            extra = {"mal_norm": float(np.linalg.norm(mal)),
                     "mean_norm": float(np.linalg.norm(mu)),
                     "sigma_mean": float(sigma.mean())}
        else:
            out = {c: bufs[c] for c in cids}
            extra = {}
        s_computed = snap()

        for c in cids:
            try:
                conns[c].send_bytes(out[c])
            except (OSError, BrokenPipeError):
                conns.pop(c, None)
        s_sent = snap()

        rec = {"event": "round", "round": rnd,
               "t_wait0": s_prev[3], "t_first_rx": s_first[3] if s_first else None,
               "t_gathered": s_gathered[3], "t_computed": s_computed[3], "t_sent": s_sent[3],
               "bytes_in": int(sum(len(b) for b in bufs.values())), **extra}
        rec.update(seg("idle", s_prev, s_first or s_gathered))
        rec.update(seg("gather", s_first or s_gathered, s_gathered))
        rec.update(seg("compute", s_gathered, s_computed))
        rec.update(seg("reply", s_computed, s_sent))
        log.write(rec)
        s_prev = s_sent

    log.write({"event": "end", "t": mono_ns()})
    log.close()
    listener.close()


if __name__ == "__main__":
    main()
