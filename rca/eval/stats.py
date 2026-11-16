#!/usr/bin/env python3
"""stats.py — statistical significance for the RQ1 main comparison.

Computes, for the single PRE-REGISTERED comparison "Ours vs the strongest
baseline", two things the prior-work corpus omits entirely:

  * Paired bootstrap 95% CI on FT Macro-F1 for BOTH systems (Ours and the
    baseline), plus a CI on their paired difference. The bootstrap resamples
    the shared test experiments (matched by run_id), recomputing sklearn
    macro-F1 over the fixed 12-class taxonomy on each resample. Seed is fixed,
    so the interval is reproducible.

  * Exact (binomial) McNemar test on per-sample FT correctness. With 72 samples
    the discordant count is small, so we use the exact binomial form
    (scipy.stats.binomtest on the discordant pairs) rather than the chi-square
    approximation.

Only this one comparison is tested — stated in the paper to avoid a
multiple-comparison objection.

Usage:
  python rca/eval/stats.py --ours results/<ours>.json --baseline results/<best_baseline>.json \
      [--system tt|sn] [--metric fault_type_macro_f1] [--n-bootstrap 10000] [--seed 0] [--output x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import score_unified as su  # noqa: E402

try:
    from sklearn.metrics import f1_score
    from scipy.stats import binomtest
except ImportError:
    sys.exit("stats.py needs scikit-learn and scipy.")


def _macro_f1(gt, pred, labels):
    return f1_score(gt, pred, labels=labels, average="macro", zero_division=0)


def _pair_by_run_id(ours_ps: list, base_ps: list) -> list:
    """Align the two runs' per-sample records on run_id; keep the shared ones."""
    base_by_id = {p["run_id"]: p for p in base_ps if p.get("run_id") is not None}
    paired = []
    for po in ours_ps:
        rid = po.get("run_id")
        pb = base_by_id.get(rid)
        if pb is not None:
            paired.append((po, pb))
    return paired


def run(ours_path, base_path, system, n_boot, seed):
    sc_o = su.score_file(ours_path, system)
    sc_b = su.score_file(base_path, system)
    system = sc_o["system"]
    if sc_b["system"] != system:
        sys.stderr.write(
            f"[stats] warning: Ours detected as {system} but baseline as "
            f"{sc_b['system']}; forcing both to {system}.\n"
        )
        sc_b = su.score_file(base_path, system)

    labels = su.FAULT_LABELS[system]
    paired = _pair_by_run_id(sc_o["per_sample"], sc_b["per_sample"])
    n = len(paired)
    if n == 0:
        sys.exit("[stats] no shared run_ids between the two files — cannot pair.")

    gt = np.array([po["gt_ft"] for po, _ in paired], dtype=object)
    pred_o = np.array([po["pred_ft"] for po, _ in paired], dtype=object)
    pred_b = np.array([pb["pred_ft"] for _, pb in paired], dtype=object)
    ok_o = np.array([bool(po["ft_correct"]) for po, _ in paired])
    ok_b = np.array([bool(pb["ft_correct"]) for _, pb in paired])

    point_o = _macro_f1(gt, pred_o, labels)
    point_b = _macro_f1(gt, pred_b, labels)

    # --- paired bootstrap ---
    rng = np.random.default_rng(seed)
    f1o = np.empty(n_boot)
    f1b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        f1o[i] = _macro_f1(gt[idx], pred_o[idx], labels)
        f1b[i] = _macro_f1(gt[idx], pred_b[idx], labels)
    diff = f1o - f1b

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    # --- exact McNemar on FT correctness ---
    b = int(np.sum(ok_o & ~ok_b))   # Ours right, baseline wrong
    c = int(np.sum(~ok_o & ok_b))   # Ours wrong, baseline right
    nd = b + c
    if nd > 0:
        p_mcnemar = float(binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue)
    else:
        p_mcnemar = 1.0

    return {
        "system": system,
        "n_paired": n,
        "n_bootstrap": n_boot,
        "seed": seed,
        "metric": "fault_type_macro_f1",
        "ours": {"point": float(point_o), "ci95": ci(f1o)},
        "baseline": {"point": float(point_b), "ci95": ci(f1b)},
        "difference_ours_minus_baseline": {
            "point": float(point_o - point_b),
            "ci95": ci(diff),
            "significant_at_0.05": not (ci(diff)[0] <= 0.0 <= ci(diff)[1]),
        },
        "mcnemar_exact": {
            "ours_right_baseline_wrong": b,
            "ours_wrong_baseline_right": c,
            "discordant_total": nd,
            "p_value": p_mcnemar,
            "significant_at_0.05": p_mcnemar < 0.05,
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Paired bootstrap CI + exact McNemar for Ours vs strongest baseline."
    )
    ap.add_argument("--ours", required=True, help="Ours result JSON")
    ap.add_argument("--baseline", required=True, help="strongest baseline result JSON")
    ap.add_argument("--system", choices=["tt", "sn"], default=None)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out = run(args.ours, args.baseline, args.system, args.n_bootstrap, args.seed)

    o, bl, d, mc = out["ours"], out["baseline"], out["difference_ours_minus_baseline"], out["mcnemar_exact"]
    print(f"\n{'='*68}")
    print(f"SIGNIFICANCE [{out['system']}]  n={out['n_paired']} paired, "
          f"{out['n_bootstrap']} bootstraps, seed={out['seed']}")
    print(f"{'='*68}")
    print(f"FT Macro-F1")
    print(f"  Ours     : {o['point']:.3f}  95% CI [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]")
    print(f"  Baseline : {bl['point']:.3f}  95% CI [{bl['ci95'][0]:.3f}, {bl['ci95'][1]:.3f}]")
    print(f"  Diff     : {d['point']:+.3f}  95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
          f"   {'(significant)' if d['significant_at_0.05'] else '(n.s.)'}")
    print(f"McNemar (exact binomial) on FT correctness")
    print(f"  Ours✓/Base✗ = {mc['ours_right_baseline_wrong']}, "
          f"Ours✗/Base✓ = {mc['ours_wrong_baseline_right']}, "
          f"discordant = {mc['discordant_total']}")
    print(f"  p = {mc['p_value']:.2e}   {'(significant)' if mc['significant_at_0.05'] else '(n.s.)'}")

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote stats → {args.output}")


if __name__ == "__main__":
    main()
