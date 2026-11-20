"""CPU topology discovery and the role -> logical-CPU plan.

Design rules (any Linux machine; topology read from /sys):

* every *role that does work* (server, each client, the ALIE colluder) gets its
  own **physical core**, and that core's SMT sibling is left idle, so no two
  roles ever share an execution pipeline, L1 or L2;
* the system monitor and the campaign orchestrator share the SMT sibling of the
  server core ("housekeeping" CPU) -- they mostly sleep;
* the Byzantine clients are spread across L3 domains (e.g. AMD CCXs) so that
  "is Byzantine" is not confounded with "shares a cache with X".

With fewer physical cores than roles the plan falls back to one logical CPU
per role, then to sharing, and says so in ``plan["warnings"]``.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

from .common import read_text


@dataclass
class LogicalCPU:
    cpu: int
    core_id: int
    package: int
    siblings: tuple[int, ...]
    l3: tuple[int, ...]


def _parse_list(s: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(sorted(out))


def logical_cpus() -> list[LogicalCPU]:
    online = _parse_list(read_text("/sys/devices/system/cpu/online", "0"))
    # deliberately NOT filtered by our own affinity mask: with isolcpus= the
    # isolated CPUs are exactly the ones we want to pin onto.
    cpus = []
    for c in online:
        base = f"/sys/devices/system/cpu/cpu{c}"
        core_id = int(read_text(f"{base}/topology/core_id", str(c)) or c)
        pkg = int(read_text(f"{base}/topology/physical_package_id", "0") or 0)
        sib = _parse_list(read_text(f"{base}/topology/thread_siblings_list", str(c)) or str(c))
        l3 = (c,)
        for idx in glob.glob(f"{base}/cache/index*"):
            if read_text(f"{idx}/level") == "3":
                l3 = _parse_list(read_text(f"{idx}/shared_cpu_list", str(c)))
        cpus.append(LogicalCPU(c, core_id, pkg, sib, l3))
    return cpus


def physical_cores(cpus: list[LogicalCPU]) -> list[tuple[int, ...]]:
    seen, cores = set(), []
    for lc in cpus:
        key = tuple(s for s in lc.siblings if s in {x.cpu for x in cpus})
        if key not in seen:
            seen.add(key)
            cores.append(key)
    return cores


def make_plan(n_clients: int, n_byz: int, override: str | None = None,
              byz_ids: str | None = None) -> dict:
    """Return {"server": cpu, "clients": [cpu...], "attacker": cpu,
    "housekeeping": cpu, "byz_ids": [...], "l3_of": {...}, "warnings": [...]}."""
    cpus = logical_cpus()
    by_cpu = {c.cpu: c for c in cpus}
    warnings: list[str] = []

    if override:
        # "server=0,clients=1:2:3:4:5:6,attacker=7,housekeeping=8"
        kv = dict(p.split("=", 1) for p in override.split(","))
        plan = {
            "server": int(kv["server"]),
            "clients": [int(x) for x in kv["clients"].split(":")],
            "attacker": int(kv["attacker"]),
            "housekeeping": int(kv.get("housekeeping", kv["server"])),
        }
        if len(plan["clients"]) != n_clients:
            raise ValueError("override lists a different number of client CPUs")
    else:
        cores = physical_cores(cpus)
        need = 1 + n_clients + 1                       # server + clients + attacker
        if len(cores) >= need:
            # tier 1: one physical core per role, SMT siblings left idle
            primary = [c[0] for c in cores]
            plan = {"server": primary[0], "clients": primary[1:1 + n_clients],
                    "attacker": primary[1 + n_clients]}
            if len(cores[0]) > 1:
                plan["housekeeping"] = cores[0][1]     # sibling of the server core
            elif len(cores) > need:
                plan["housekeeping"] = primary[need]   # a spare core
            else:
                plan["housekeeping"] = primary[0]
                warnings.append("no spare CPU: monitor shares the server core")
        elif len(cpus) >= need:
            # tier 2: one logical CPU per role; first threads of every core, then siblings
            order = [c[i] for i in range(max(len(c) for c in cores)) for c in cores if i < len(c)]
            plan = {"server": order[0], "clients": order[1:1 + n_clients],
                    "attacker": order[1 + n_clients],
                    "housekeeping": order[need] if len(order) > need else order[0]}
            warnings.append(f"only {len(cores)} physical cores for {need} roles: some roles are "
                            "SMT siblings (they share L1/L2 and execution units)")
        else:
            # tier 3: not enough CPUs at all -- share round-robin
            prim = [c.cpu for c in cpus] or [0]
            rr = lambda i: prim[i % len(prim)]
            plan = {"server": rr(0), "clients": [rr(1 + i) for i in range(n_clients)],
                    "attacker": rr(1 + n_clients), "housekeeping": rr(0)}
            warnings.append(f"only {len(prim)} CPUs for {need} roles: roles share CPUs "
                            "(fine for a smoke test, NOT for measurement; try fewer --n-clients)")

    l3_of = {c: (by_cpu[c].l3[0] if c in by_cpu else -1)
             for c in [plan["server"], plan["attacker"], *plan["clients"]]}

    if byz_ids and byz_ids != "auto":
        chosen = [int(x) for x in byz_ids.split(",") if x != ""]
    else:
        # spread Byzantine clients across L3 domains, round-robin
        domains: dict[int, list[int]] = {}
        for cid, cpu in enumerate(plan["clients"]):
            domains.setdefault(l3_of[cpu], []).append(cid)
        order = []
        lists = list(domains.values())
        i = 0
        while any(lists):
            lst = lists[i % len(lists)]
            if lst:
                order.append(lst.pop(0))
            i += 1
            if i > 10_000:
                break
        # take every k-th so the picks are not adjacent within a CCX either
        chosen = order[:n_byz]
    if len(chosen) != n_byz:
        raise ValueError(f"need {n_byz} byzantine ids, got {chosen}")

    plan["byz_ids"] = sorted(chosen)
    plan["l3_of"] = {str(k): v for k, v in l3_of.items()}
    plan["warnings"] = warnings
    used = {plan["server"], plan["attacker"], plan["housekeeping"], *plan["clients"]}
    plan["idle_cpus"] = sorted(set(by_cpu) - used)
    plan["role_cpus"] = sorted(used)
    return plan


def describe(plan: dict) -> str:
    lines = [f"server      -> cpu {plan['server']}"]
    for i, c in enumerate(plan["clients"]):
        tag = "BYZ" if i in plan["byz_ids"] else "   "
        lines.append(f"client {i} {tag} -> cpu {c}  (L3 group {plan['l3_of'][str(c)]})")
    lines.append(f"attacker    -> cpu {plan['attacker']}")
    lines.append(f"monitor/orch-> cpu {plan['housekeeping']}")
    lines.append(f"idle        -> {plan['idle_cpus']}")
    for w in plan["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    p = make_plan(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
                  int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    print(describe(p))
    print(json.dumps({k: v for k, v in p.items() if k != "topology"}, indent=1))
