"""Turn a campaign into feature tables and answer: is there a hardware signature?

    python -m hwsig.analyze results/alie_v1 [--skip-rounds 1]

Outputs to <campaign>/analysis/:

  client_rounds.parquet|csv   one row per (run, round, client): in-process OS +
                              perf counters per segment, monitor features of the
                              client's core over its fit window, model.* metrics
  rounds.parquet|csv          one row per (run, round): server, attacker core,
                              system-wide, and colluder / non-colluder client means
  univariate_rounds.csv       per feature: honest vs attack at run level
                              (Mann-Whitney U, AUC, Cohen's d, paired Wilcoxon, BH q)
  univariate_colluders.csv    same, restricted to the designated colluder clients
  univariate_honest_clients.csv  same, non-colluder clients (spill-over test)
  classifiers.csv             grouped-CV AUC per (question, view, feature class, model)
  importance_*.csv            top features of the best models
  report.md                   human summary
  *.png                       plots (if matplotlib is installed)

Feature classes (by name): ``hw`` = PMU events and their ratios, frequency,
C-state residency, temperatures, energy/power; ``os`` = scheduler, rusage,
/proc, IRQ, memory, timing; ``model`` = update/loss statistics (the classic
defence baseline, kept separate on purpose).
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_records

warnings.filterwarnings("ignore")

HW_EVENTS = ("cycles", "instructions", "cache_references", "cache_misses", "branches",
             "branch_misses", "stalled_cycles", "ref_cycles", "l1d_", "l1i_", "llc_", "dtlb_",
             "itlb_", "bpu_", "ipc", "miss_rate", "stall_frac", "ghz_eff")
HW_MON = ("freq_khz", "cstate", "hwmon_", "rapl_", "msr_", "psu_", "perf_", "_J", "_W",
          "power", "energy", "temp")


def feature_class(name: str) -> str:
    if name.startswith("model.") or ".model." in name:
        return "model"
    low = name.lower()
    if any(e in low for e in HW_EVENTS) and (".thr." in low or ".core." in low or "perf_" in low):
        return "hw"
    if any(e.lower() in low for e in HW_MON):
        return "hw"
    return "os"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


GAUGE_PAT = re.compile(r"(freq_khz|^hwmon_(?!.*energy)|psu_.*(power_now|current_now|voltage_now|"
                       r"capacity|online)|^mem_|loadavg|procs_running|procs_blocked|CurrEstab|"
                       r"sample_cost)")


def _unwrap(v: np.ndarray, modulus: float) -> np.ndarray:
    d = np.diff(v, prepend=v[:1])
    d[d < 0] += modulus
    return v[0] + np.cumsum(d) - d[0]


def load_monitor(rdir: Path):
    p = rdir / "monitor.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, low_memory=False)
    meta = json.loads((rdir / "monitor_meta.json").read_text()) if (rdir / "monitor_meta.json").exists() else {}
    for c, v in meta.get("constants", {}).items():    # put run-constant columns back
        try:
            df[c] = float(v)
        except ValueError:
            pass
    df = df.dropna(axis=1, how="all").sort_values("t_mono_ns").reset_index(drop=True)
    df = df.ffill().bfill()
    unit = meta.get("msr_energy_unit_J")
    for c in [c for c in df.columns if c.startswith("msr_") and c.endswith("_raw")]:
        v = df[c].to_numpy(dtype=np.float64)
        df[c.replace("_raw", "_J")] = _unwrap(v, 2.0 ** 32) * (unit or 1.0)
        df.drop(columns=c, inplace=True)
    for c in [c for c in df.columns if c.startswith("rapl_")]:
        mx = meta.get("rapl_max_range_uj", {}).get(c)
        v = df[c].to_numpy(dtype=np.float64)
        df[c] = _unwrap(v, float(mx) if mx else 2.0 ** 32) / 1e6
        df.rename(columns={c: c[:-3] + "_J"}, inplace=True)
    for c in [c for c in df.columns if c.startswith("hwmon_") and "_energy_" in c]:
        df[c] = df[c].astype(np.float64) / 1e6  # uJ -> J (amd_energy)
    perf = load_perf_cli(rdir, meta)
    return df, perf, meta


def load_perf_cli(rdir: Path, meta: dict):
    p = rdir / "perf_cli.csv"
    if not p.exists() or "perf_cli_start_mono_ns" not in meta:
        return None
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 5:
            continue
        try:
            t = float(f[0])
        except ValueError:
            continue
        cpu = f[1].strip()
        m = re.match(r"CPU(\d+)", cpu)
        if not m:
            continue
        try:
            val = float(f[2])
        except ValueError:
            val = np.nan
        ev = re.sub(r"[^A-Za-z0-9]+", "_", f[4]).strip("_")
        rows.append((t, int(m.group(1)), ev, val))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["t", "cpu", "ev", "v"])
    d["col"] = "cpu" + d["cpu"].astype(str) + "_perf_" + d["ev"]
    w = d.pivot_table(index="t", columns="col", values="v", aggfunc="sum").sort_index()
    w = w.fillna(0).cumsum()  # interval counts -> cumulative
    t0 = (meta["perf_cli_start_mono_ns"] + meta.get("perf_cli_start_mono_ns_after",
                                                     meta["perf_cli_start_mono_ns"])) / 2
    w.insert(0, "t_mono_ns", (t0 + w.index.to_numpy() * 1e9).astype(np.int64))
    first = {c: 0.0 for c in w.columns if c != "t_mono_ns"}
    head = pd.DataFrame([{"t_mono_ns": int(t0), **first}])
    return pd.concat([head, w.reset_index(drop=True)], ignore_index=True)


class Windower:
    """Fast window statistics on a monitor table."""

    def __init__(self, df: pd.DataFrame):
        self.t = df["t_mono_ns"].to_numpy(dtype=np.float64)
        self.cols = [c for c in df.columns if c != "t_mono_ns"]
        self.V = df[self.cols].to_numpy(dtype=np.float64)
        self.gauge = np.array([bool(GAUGE_PAT.search(c)) for c in self.cols])

    def _interp(self, x: float) -> np.ndarray:
        t = self.t
        if len(t) == 1:
            return self.V[0]
        i = int(np.clip(np.searchsorted(t, x), 1, len(t) - 1))
        w = (x - t[i - 1]) / max(t[i] - t[i - 1], 1.0)
        w = min(max(w, 0.0), 1.0)
        return self.V[i - 1] + w * (self.V[i] - self.V[i - 1])

    def window(self, t0: float, t1: float) -> dict:
        a, b = self._interp(t0), self._interp(t1)
        delta = b - a
        m = (self.t >= t0) & (self.t <= t1)
        gmean = np.nanmean(self.V[m], axis=0) if m.sum() else (a + b) / 2
        gmax = np.nanmax(self.V[m], axis=0) if m.sum() else np.maximum(a, b)
        out = {}
        dt = max((t1 - t0) / 1e9, 1e-9)
        for j, c in enumerate(self.cols):
            if self.gauge[j]:
                out[c] = gmean[j]
                if "temp" in c or "freq" in c:
                    out[c + "__max"] = gmax[j]
            else:
                out[c] = delta[j]
                if c.endswith("_J"):
                    out[c[:-2] + "_W"] = delta[j] / dt
        out["__dt_s"] = dt
        return out


def per_cpu(win: dict, cpu: int, prefix: str) -> dict:
    """Pick the columns that belong to one logical CPU and rename them."""
    tag = f"cpu{cpu}_"
    return {prefix + k[len(tag):]: v for k, v in win.items() if k.startswith(tag)}


def system_only(win: dict, prefix: str) -> dict:
    return {prefix + k: v for k, v in win.items() if not re.match(r"cpu\d+_", k)}


def fix_msr_names(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        m = re.match(r"msr_core_energy_J_cpu(\d+)$", c)
        if m:
            ren[c] = f"cpu{m.group(1)}_msr_core_energy_J"
    return df.rename(columns=ren)


def add_derived(row: dict) -> dict:
    """IPC, miss rates, stall fractions, effective GHz, utilisation, power."""
    add = {}
    for k in list(row):
        if k.endswith(".cycles") or k.endswith("_perf_cycles"):
            p = k[: -len("cycles")]
            cyc = row.get(k)
            if not cyc or not np.isfinite(cyc):
                continue
            ins = row.get(p + "instructions")
            if ins is not None:
                add[p + "ipc"] = ins / cyc
            for s in ("stalled_cycles_backend", "stalled_cycles_frontend"):
                if row.get(p + s) is not None:
                    add[p + s.replace("stalled_cycles_", "stall_frac_")] = row[p + s] / cyc
            tc = row.get(p + "task_clock_ns") or row.get(p + "cpu_clock_ns")
            if tc:
                add[p + "ghz_eff"] = cyc / tc
        for num, den, name in (("cache_misses", "cache_references", "cache_miss_rate"),
                               ("branch_misses", "branches", "branch_miss_rate"),
                               ("l1d_load_misses", "l1d_loads", "l1d_miss_rate"),
                               ("l1i_load_misses", "l1i_loads", "l1i_miss_rate"),
                               ("llc_load_misses", "llc_loads", "llc_miss_rate"),
                               ("dtlb_load_misses", "dtlb_loads", "dtlb_miss_rate"),
                               ("itlb_load_misses", "itlb_loads", "itlb_miss_rate"),
                               ("bpu_load_misses", "bpu_loads", "bpu_miss_rate")):
            if k.endswith("." + num) or k.endswith("_" + num):
                p = k[: -len(num)]
                d = row.get(p + den)
                if d:
                    add[p + name] = row[k] / d
        if k.endswith("perf_msr_aperf"):
            p = k[: -len("msr_aperf")]
            mp = row.get(p + "msr_mperf")
            if mp:
                add[p + "aperf_mperf_ratio"] = row[k] / mp
        if k.endswith("os.thread_cpu_ns"):
            p = k[: -len("thread_cpu_ns")]
            w = row.get(p + "wall_ns")
            if w:
                add[p + "thread_util"] = row[k] / w
    row.update(add)
    return row


def stat_busy(row: dict, prefix: str) -> None:
    keys = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"]
    vals = [row.get(f"{prefix}stat_{k}") for k in keys]
    if all(v is not None for v in vals):
        tot = sum(vals)
        if tot > 0:
            row[f"{prefix}busy_frac"] = 1 - (vals[3] + vals[4]) / tot


# --------------------------------------------------------------------------- #
def build_tables(root: Path):
    camp = json.loads((root / "campaign.json").read_text())
    plan = camp["plan"]
    client_rows, round_rows, runs = [], [], []
    for rdir in sorted((root / "runs").glob("*")):
        if not (rdir / "DONE").exists():
            continue
        meta = json.loads((rdir / "meta.json").read_text())
        cfg = {**camp["args"], **meta["cfg"]}
        mode, run_id, pair = cfg["mode"], cfg["run_id"], cfg["pair"]
        colluders = set(meta.get("colluder_ids", plan["byz_ids"]))
        designated = set(plan["byz_ids"])
        mon = load_monitor(rdir)
        W = Wp = None
        if mon is not None:
            df, perf, mmeta = mon
            df = fix_msr_names(df)
            W = Windower(df)
            Wp = Windower(perf) if perf is not None else None
        srv = {r["round"]: r for r in read_records(rdir / "server.jsonl", "round")}
        att = {r["round"]: r for r in read_records(rdir / "attacker.jsonl", "round")}
        fits = {}
        for cid in range(cfg["n_clients"]):
            for r in read_records(rdir / f"client_{cid}.jsonl", "fit"):
                fits[(r["round"], cid)] = r
        final_acc = srv[max(srv)]["test_acc"] if srv else np.nan
        runs.append({"run_id": run_id, "pair": pair, "mode": mode, "final_acc": final_acc,
                     "temp_pre_c": meta.get("temp_pre_c"), "temp_post_c": meta.get("temp_post_c"),
                     **{f"acc_r{k}": v["test_acc"] for k, v in srv.items()}})

        def win(w, t0, t1):
            return w.window(t0, t1) if w is not None else {}

        for rnd in sorted(k for k in srv if k > 0):
            s = srv[rnd]
            base = {"run_id": run_id, "pair": pair, "mode": mode, "y": int(mode == "attack"),
                    "round": rnd}
            # ---- clients ----
            crow_list = []
            for cid, cpu in enumerate(plan["clients"]):
                f = fits.get((rnd, cid))
                if f is None:
                    continue
                row = dict(base, cid=cid, cpu=cpu, designated=int(cid in designated),
                           is_byz=int(cid in designated and mode == "attack"))
                for k, v in f.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) \
                            and not k.startswith("t_") and k not in ("cid", "cpu", "round") \
                            and "mux_min" not in k:
                        row[k] = float(v)
                row["fit.wall_ns"] = float(f["t_fit1"] - f["t_fit0"])
                for segname, t0, t1 in (("fit", f["t_fit0"], f["t_fit1"]),
                                        ("train", f["t_fit0"], f["t_train1"])):
                    wv = win(W, t0, t1)
                    row.update(per_cpu(wv, cpu, f"{segname}.mon."))
                    if Wp is not None:
                        row.update(per_cpu(win(Wp, t0, t1), cpu, f"{segname}.mon."))
                    stat_busy(row, f"{segname}.mon.")
                add_derived(row)
                client_rows.append(row)
                crow_list.append(row)

            # ---- round level ----
            rr = dict(base)
            t0, t1 = s.get("t_cfg"), s.get("t_agg1")
            for k, v in s.items():
                if isinstance(v, (int, float)) and not k.startswith("t_") and k not in (
                        "round", "n_failures") and "mux_min" not in k:
                    rr[f"server.{k}"] = float(v)
            rr.pop("server.test_acc", None)
            rr.pop("server.test_loss", None)
            if t0 and t1:
                wv = win(W, t0, t1)
                rr.update(system_only(wv, "sys."))
                rr.update(per_cpu(wv, plan["server"], "srvcore.mon."))
                rr.update(per_cpu(wv, plan["attacker"], "attcore.mon."))
                if Wp is not None:
                    wp = win(Wp, t0, t1)
                    rr.update(per_cpu(wp, plan["server"], "srvcore.mon."))
                    rr.update(per_cpu(wp, plan["attacker"], "attcore.mon."))
                    rr.update(system_only(wp, "sys."))
                stat_busy(rr, "srvcore.mon.")
                stat_busy(rr, "attcore.mon.")
            a = att.get(rnd)
            if a:
                for k, v in a.items():
                    if isinstance(v, (int, float)) and not k.startswith("t_") and k != "round" \
                            and "mux_min" not in k and k not in ("mal_norm", "mean_norm", "sigma_mean"):
                        rr[f"attacker.{k}"] = float(v)
            if crow_list:
                cdf = pd.DataFrame(crow_list)
                num = [c for c in cdf.columns if c not in base and c not in
                       ("cid", "cpu", "designated", "is_byz")]
                for grp, mask in (("colluders", cdf["designated"] == 1),
                                  ("others", cdf["designated"] == 0)):
                    if mask.any():
                        m = cdf.loc[mask, num].mean(numeric_only=True)
                        rr.update({f"{grp}.{k}": v for k, v in m.items()})
                fit_ends = [r["fit.wall_ns"] for r in crow_list if "fit.wall_ns" in r]
                if fit_ends:
                    rr["round.client_fit_spread_ns"] = float(np.max(fit_ends) - np.min(fit_ends))
            add_derived(rr)
            stat_busy(rr, "sys.")
            round_rows.append(rr)
    return pd.DataFrame(client_rows), pd.DataFrame(round_rows), pd.DataFrame(runs), camp


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"run_id", "pair", "mode", "y", "round", "cid", "cpu", "designated", "is_byz"}
    cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    good = []
    for c in cols:
        v = df[c]
        if v.isna().mean() > 0.5:
            continue
        if v.nunique(dropna=True) <= 1:
            continue
        good.append(c)
    return good


def univariate(df: pd.DataFrame, unit_col: str = "run_id") -> pd.DataFrame:
    """Aggregate to one value per run (rounds are not independent), then test."""
    from scipy import stats
    feats = feature_cols(df)
    agg = df.groupby([unit_col, "pair", "mode"])[feats].mean().reset_index()
    h = agg[agg["mode"] == "honest"]
    a = agg[agg["mode"] == "attack"]
    paired = h.set_index("pair")[feats].join(a.set_index("pair")[feats], lsuffix="_h",
                                             rsuffix="_a", how="inner")
    out = []
    for c in feats:
        x, y = h[c].dropna().to_numpy(), a[c].dropna().to_numpy()
        if len(x) < 2 or len(y) < 2:
            continue
        try:
            u, p = stats.mannwhitneyu(y, x, alternative="two-sided")
            auc = u / (len(x) * len(y))
        except ValueError:
            p, auc = np.nan, np.nan
        sd = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
                     / max(len(x) + len(y) - 2, 1))
        d = (y.mean() - x.mean()) / sd if sd > 0 else np.nan
        pw = np.nan
        if len(paired) >= 3:
            dd = (paired[c + "_a"] - paired[c + "_h"]).dropna()
            if len(dd) >= 3 and (dd != 0).any():
                try:
                    pw = stats.wilcoxon(dd).pvalue
                except ValueError:
                    pass
        out.append({"feature": c, "class": feature_class(c), "n_honest": len(x),
                    "n_attack": len(y), "mean_honest": x.mean(), "mean_attack": y.mean(),
                    "rel_diff": (y.mean() - x.mean()) / (abs(x.mean()) + 1e-30),
                    "cohen_d": d, "auc": auc, "p_mwu": p, "p_wilcoxon_paired": pw})
    r = pd.DataFrame(out)
    if r.empty:
        return r
    r["q_mwu"] = bh(r["p_mwu"].to_numpy())
    r["q_wilcoxon"] = bh(r["p_wilcoxon_paired"].to_numpy())
    r["auc_dev"] = (r["auc"] - 0.5).abs()
    return r.sort_values(["auc_dev", "p_mwu"], ascending=[False, True]).reset_index(drop=True)


VIEWS = {
    "all": lambda c: True,
    "no_attacker_core": lambda c: not (c.startswith("attacker.") or c.startswith("attcore.")),
    "defender_view": lambda c: c.startswith(("server.", "srvcore.", "others.")),
    "colluders_only": lambda c: c.startswith("colluders."),
    "system_wide_only": lambda c: c.startswith("sys."),
}
CLASSES = {
    "hw": lambda c: feature_class(c) == "hw",
    "os": lambda c: feature_class(c) == "os",
    "hw+os": lambda c: feature_class(c) in ("hw", "os"),
    "model": lambda c: feature_class(c) == "model",
}


def cv_auc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, run_ids: np.ndarray,
           model: str, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ug = np.unique(groups)
    if len(ug) < 2 or len(np.unique(y)) < 2:
        return None
    k = min(5, len(ug))
    proba = np.full(len(y), np.nan)
    imps = []
    Xv = np.array(X.to_numpy(dtype=np.float64), copy=True)
    Xv[~np.isfinite(Xv)] = np.nan
    for tr, te in GroupKFold(n_splits=k).split(Xv, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        if model == "logreg":
            clf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                StandardScaler(),
                                LogisticRegression(C=0.1, penalty="l2", max_iter=2000,
                                                   class_weight="balanced"))
        else:
            clf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                       max_features="sqrt", n_jobs=-1,
                                                       class_weight="balanced",
                                                       random_state=seed))
        clf.fit(Xv[tr], y[tr])
        proba[te] = clf.predict_proba(Xv[te])[:, 1]
        last = clf[-1]
        imps.append(np.abs(last.coef_[0]) if model == "logreg" else last.feature_importances_)
    ok = np.isfinite(proba)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return None
    auc = roc_auc_score(y[ok], proba[ok])
    rd = pd.DataFrame({"run": run_ids[ok], "y": y[ok], "p": proba[ok]}).groupby("run").mean()
    run_auc = roc_auc_score(rd["y"], rd["p"]) if rd["y"].nunique() == 2 else np.nan
    imp = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
    return auc, run_auc, imp


def null_auc(X, y, groups, run_ids, n_perm: int, seed: int = 0):
    """Permutation null for the logreg AUC: shuffle labels at the *run* level."""
    rng = np.random.default_rng(seed)
    runs = pd.Series(y, index=run_ids).groupby(level=0).first()
    vals = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(runs.to_numpy()), index=runs.index)
        yp = perm.loc[run_ids].to_numpy()
        r = cv_auc(X, yp, groups, run_ids, "logreg", seed)
        if r:
            vals.append(r[0])
    return np.array(vals)


def classify(df: pd.DataFrame, question: str, group_col: str, views, n_perm: int):
    feats = feature_cols(df)
    y = df["y"].to_numpy()
    g = df[group_col].to_numpy()
    runs = df["run_id"].to_numpy()
    res, imps = [], {}
    for vname, vf in views.items():
        for cname, cf in CLASSES.items():
            cols = [c for c in feats if vf(c) and cf(c)]
            if len(cols) < 1:
                continue
            for model in ("logreg", "rf"):
                r = cv_auc(df[cols], y, g, runs, model)
                if r is None:
                    continue
                auc, run_auc, imp = r
                row = {"question": question, "view": vname, "features": cname, "model": model,
                       "n_features": len(cols), "n_rows": len(df), "auc_round": auc,
                       "auc_run": run_auc}
                if n_perm and model == "logreg" and cname in ("hw", "hw+os"):
                    nul = null_auc(df[cols], y, g, runs, n_perm)
                    if len(nul):
                        row["perm_p"] = (1 + (nul >= auc).sum()) / (1 + len(nul))
                        row["null_auc_mean"] = nul.mean()
                res.append(row)
                imps[(question, vname, cname, model)] = imp.head(40)
    return pd.DataFrame(res), imps


# --------------------------------------------------------------------------- #
def plots(out: Path, runs: pd.DataFrame, rounds: pd.DataFrame, uni: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    acc_cols = sorted([c for c in runs.columns if c.startswith("acc_r")], key=lambda s: int(s[5:]))
    if acc_cols:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for mode, col in (("honest", "#2a6fdb"), ("attack", "#d1495b")):
            sub = runs[runs["mode"] == mode][acc_cols]
            if sub.empty:
                continue
            xs = [int(c[5:]) for c in acc_cols]
            for _, r in sub.iterrows():
                ax.plot(xs, r.to_numpy(), color=col, alpha=0.25, lw=0.8)
            ax.plot(xs, sub.mean().to_numpy(), color=col, lw=2.2, label=f"{mode} (mean)")
        ax.set_xlabel("round")
        ax.set_ylabel("test accuracy")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "accuracy.png", dpi=130)
        plt.close(fig)
        made.append("accuracy.png")
    if not uni.empty:
        top = uni[uni["class"] != "model"].head(12)["feature"].tolist()
        if top:
            agg = rounds.groupby(["run_id", "mode"])[top].mean().reset_index()
            n = len(top)
            ncol = 4
            nrow = int(np.ceil(n / ncol))
            fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
            for i, f in enumerate(top):
                ax = axs[i // ncol][i % ncol]
                data = [agg[agg["mode"] == m][f].dropna() for m in ("honest", "attack")]
                ax.boxplot(data, tick_labels=["honest", "attack"], widths=0.5)
                for j, d in enumerate(data):
                    ax.scatter(np.full(len(d), j + 1) + np.random.uniform(-.08, .08, len(d)),
                               d, s=10, alpha=0.6, color=("#2a6fdb", "#d1495b")[j])
                ax.set_title(f if len(f) < 42 else "…" + f[-40:], fontsize=7)
                ax.tick_params(labelsize=7)
            for j in range(n, nrow * ncol):
                axs[j // ncol][j % ncol].axis("off")
            fig.tight_layout()
            fig.savefig(out / "top_features.png", dpi=130)
            plt.close(fig)
            made.append("top_features.png")
    return made


def md_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return "_none_\n"
    d = df[cols].head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.3g}" if pd.notna(v) else "")
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.to_numpy()) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--skip-rounds", type=int, default=1, help="drop warm-up rounds <= this")
    ap.add_argument("--perm", type=int, default=0, help="label permutations for null AUC")
    a = ap.parse_args(argv)

    root = Path(a.campaign)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    clients, rounds, runs, camp = build_tables(root)
    if rounds.empty:
        raise SystemExit("no completed runs found")
    for name, d in (("client_rounds", clients), ("rounds", rounds), ("runs", runs)):
        d.to_csv(out / f"{name}.csv", index=False)
        try:
            d.to_parquet(out / f"{name}.parquet", index=False)
        except Exception:  # noqa: BLE001
            pass

    r_use = rounds[rounds["round"] > a.skip_rounds]
    c_use = clients[clients["round"] > a.skip_rounds]

    uni_r = univariate(r_use)
    uni_c = univariate(c_use[c_use["designated"] == 1])
    uni_h = univariate(c_use[c_use["designated"] == 0])
    uni_r.to_csv(out / "univariate_rounds.csv", index=False)
    uni_c.to_csv(out / "univariate_colluders.csv", index=False)
    uni_h.to_csv(out / "univariate_honest_clients.csv", index=False)

    results, all_imps = [], {}
    r1, i1 = classify(r_use, "Q1 round: attack vs honest", "pair", VIEWS, a.perm)
    results.append(r1); all_imps.update(i1)
    own = {"client_own": lambda c: True}
    att = c_use[c_use["mode"] == "attack"].copy()
    att["y"] = att["is_byz"]
    r2, i2 = classify(att, "Q2 client: which client is Byzantine (attack runs)", "run_id", own, 0)
    results.append(r2); all_imps.update(i2)
    col = c_use[c_use["designated"] == 1]
    r3, i3 = classify(col, "Q3 colluder client: attack vs shadow", "pair", own, a.perm)
    results.append(r3); all_imps.update(i3)
    hon = c_use[c_use["designated"] == 0]
    r4, i4 = classify(hon, "Q4 honest client: attack vs honest (spill-over)", "pair", own, 0)
    results.append(r4); all_imps.update(i4)
    res = pd.concat([r for r in results if not r.empty], ignore_index=True) if any(
        not r.empty for r in results) else pd.DataFrame()
    res.to_csv(out / "classifiers.csv", index=False)

    imp_rows = []
    for (q, v, c, m), s in all_imps.items():
        for rank, (f, val) in enumerate(s.items()):
            imp_rows.append({"question": q, "view": v, "features": c, "model": m,
                             "rank": rank + 1, "feature": f, "importance": val})
    pd.DataFrame(imp_rows).to_csv(out / "importances.csv", index=False)
    figs = plots(out, runs, r_use, uni_r)

    # ---------------- report ----------------
    L = [f"# Hardware-signature analysis — `{root.name}`\n"]
    n_h = (runs["mode"] == "honest").sum()
    n_a = (runs["mode"] == "attack").sum()
    L.append(f"Runs: **{n_h} honest**, **{n_a} ALIE**. Rounds analysed: > {a.skip_rounds}. "
             f"Plan: server cpu {camp['plan']['server']}, clients {camp['plan']['clients']}, "
             f"attacker cpu {camp['plan']['attacker']}, colluders {camp['plan']['byz_ids']}. "
             f"Shadow control: {not camp['args'].get('no_shadow', False)}.\n")
    for w in camp["plan"].get("warnings", []):
        L.append(f"> **Warning:** {w}\n")
    L.append("## 0. Did the attack work?\n")
    acc = runs.groupby("mode")["final_acc"].agg(["mean", "std", "count"]).reset_index()
    L.append(md_table(acc, ["mode", "mean", "std", "count"]))
    if "accuracy.png" in figs:
        L.append("![accuracy](accuracy.png)\n")
    L.append("## 1. Classifiers (grouped CV; groups = seed pair or run)\n")
    L.append("AUC 0.5 = no signal. `auc_run` averages round-level probabilities per run. "
             "`perm_p` (if computed) is a run-level label-permutation p-value.\n")
    if not res.empty:
        cols = ["question", "view", "features", "model", "n_features", "auc_round", "auc_run"]
        cols += [c for c in ("perm_p",) if c in res.columns]
        L.append(md_table(res.sort_values(["question", "auc_round"], ascending=[True, False]),
                          cols, n=200))
    L.append("## 2. Strongest single features (run-level, honest vs attack)\n")
    L.append("Each feature averaged per run, then Mann-Whitney U across runs; q = BH-FDR.\n")
    for title, u in (("Round / system level", uni_r), ("Colluder clients", uni_c),
                     ("Honest clients (spill-over)", uni_h)):
        L.append(f"### {title}\n")
        for cls in ("hw", "os", "model"):
            sub = u[u["class"] == cls] if not u.empty else u
            L.append(f"**{cls}**\n\n")
            L.append(md_table(sub, ["feature", "mean_honest", "mean_attack", "rel_diff",
                                    "cohen_d", "auc", "p_mwu", "q_mwu"], n=12))
    if "top_features.png" in figs:
        L.append("![top features](top_features.png)\n")
    L.append("## 3. How to read this\n")
    L.append(
        "* **Q1 / view=all** includes the attacker's own core. Under shadow control the only "
        "difference there is the ALIE arithmetic, so a signal confined to `attcore.*`/`attacker.*` "
        "means *the attacker's machine* is detectable, not that the federation is.\n"
        "* **Q1 / defender_view** is what honest participants (server + honest clients) can "
        "observe about themselves. A signal here would be the interesting result.\n"
        "* **Q3** asks whether a colluding client's core looks different when its update is "
        "actually replaced versus passed through.\n"
        "* **Q4** tests spill-over: the poisoned global model changes what honest clients compute "
        "next round; data-dependent hardware effects (e.g. denormals, branch behaviour in ReLU) "
        "would show up here.\n"
        "* Compare every hw/os AUC with the `model` row: that is the classical update-statistics "
        "detector, i.e. the bar a hardware signature has to clear to be useful.\n"
        "* With few runs, treat AUCs near 1 with suspicion: check `perm_p` (rerun with "
        "`--perm 200`) and the univariate q-values.\n")
    (out / "report.md").write_text("\n".join(L))
    print(f"wrote {out}/report.md")
    if not res.empty:
        print(res.sort_values("auc_round", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
