#!/usr/bin/env python3
"""Compare results/*.csv against the baseline run and report attack impact.
Usage: python3 analysis/compare_runs.py [--results results] [--baseline baseline] [--plot]"""
import argparse
import csv
import glob
import os


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if int(r["round"]) > 0]
    return [(int(r["round"]), float(r["loss"]), float(r["accuracy"])) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--last-k", type=int, default=5, help="rounds averaged for 'final' accuracy")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    runs = {os.path.basename(p)[:-4]: load(p) for p in sorted(glob.glob(f"{args.results}/*.csv"))}
    if args.baseline not in runs:
        raise SystemExit(f"baseline run '{args.baseline}.csv' not found in {args.results}/")

    def final(r): return sum(a for _, _, a in r[-args.last_k:]) / len(r[-args.last_k:])
    base = runs[args.baseline]
    b_final, b_best = final(base), max(a for _, _, a in base)
    target = 0.9 * b_best

    def rounds_to(r):
        return next((rd for rd, _, a in r if a >= target), None)

    print(f"{'run':<28}{'final_acc':>10}{'best_acc':>10}{'Δfinal_abs':>12}{'Δfinal_rel%':>13}"
          f"{'final_loss':>11}{'rnds→90%base':>14}")
    for name, r in runs.items():
        f, b = final(r), max(a for _, _, a in r)
        rt = rounds_to(r)
        print(f"{name:<28}{f:>10.4f}{b:>10.4f}{f - b_final:>+12.4f}{100 * (f - b_final) / b_final:>+13.1f}"
              f"{r[-1][1]:>11.4f}{(rt if rt is not None else 'never'):>14}")

    if args.plot:
        import matplotlib.pyplot as plt
        for name, r in runs.items():
            plt.plot([x[0] for x in r], [x[2] for x in r], label=name)
        plt.xlabel("round"); plt.ylabel("test accuracy"); plt.legend(); plt.grid(alpha=.3)
        plt.savefig(f"{args.results}/comparison.png", dpi=150, bbox_inches="tight")
        print(f"saved {args.results}/comparison.png")


if __name__ == "__main__":
    main()
