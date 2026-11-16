#!/usr/bin/env python3
"""score_unified.py — the single source of truth for the paper's headline numbers.

Re-scores ANY diagnose.py / baseline / ablation result JSON under ONE consistent
definition so that Ours, every baseline, and every ablation are directly
comparable. It removes two inconsistencies that exist between the producers:

  * Macro-F1 label set. diagnose.py averaged F1 over the gt∪pred labels actually
    seen in a run, while the baselines averaged over gt-only labels — different
    denominators, not comparable. Here we ALWAYS compute sklearn macro-F1 over
    the FIXED 12-class taxonomy of the system, so the class set is identical for
    every method.
  * Per-sample schema. Ours is nested (ground_truth / prediction blocks); the
    baselines are flat (gt_ft / pred_ft / gt_layer / pred_layer). Both are
    normalized to one record shape here.

It also emits the SRE triage-utility three-state decomposition that the
Evaluation narrative needs:

    exact diagnosis        = fault type correct
    partial triage utility = fault type wrong BUT layer correct (search space
                             still narrowed to the right layer)
    wrong direction        = layer also wrong
    triage-useful rate     = exact + partial
                           = ft_correct OR (ft_wrong AND layer_correct)

Subcommands:
  score   (default)  Layer/FT accuracy, FT macro-P/R/F1, layer macro-F1, three-state.
  reuse              Retrieval-reuse metrics from the (nested-only)
                     retrieval.neighbors: Same-FT / Same-layer Hit@K, MRR, and the
                     retrieval helps / misleads rate.

Usage:
  python rca/eval/score_unified.py score results/<run>.json [--system tt|sn] [--output x.json]
  python rca/eval/score_unified.py reuse results/<ours_run>.json [--system tt|sn] [--output x.json]
  python rca/eval/score_unified.py results/<run>.json          # 'score' is the default
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fullstack_rca.data.dataset import (  # noqa: E402
    FAULT_TYPE_TO_ID,
    SN_FAULT_TYPE_TO_ID,
)

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
except ImportError:
    sys.exit("score_unified.py needs scikit-learn (pip install scikit-learn).")

# Fixed per-system class sets — the macro-F1 denominator. Imported, never hardcoded.
FAULT_LABELS = {"tt": list(FAULT_TYPE_TO_ID), "sn": list(SN_FAULT_TYPE_TO_ID)}
LAYER_LABELS = ["L1", "L2", "L3", "L4"]
HIT_KS = (1, 3, 5)

# Classes unique to one system — used to auto-detect tt vs sn from the gt labels.
_SN_ONLY = set(SN_FAULT_TYPE_TO_ID) - set(FAULT_TYPE_TO_ID)
_TT_ONLY = set(FAULT_TYPE_TO_ID) - set(SN_FAULT_TYPE_TO_ID)


def _load_results(path: str) -> list:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data.get("results", [])


def _normalize(results: list) -> list:
    """Map nested (Ours) and flat (baseline) per-sample records to one shape."""
    if not results:
        return []
    nested = "ground_truth" in (results[0] or {})
    norm = []
    for r in results:
        if nested:
            gt = r.get("ground_truth", {}) or {}
            pr = r.get("prediction", {}) or {}
            rec = {
                "run_id": r.get("run_id"),
                "gt_ft": gt.get("fault_type"),
                "pred_ft": pr.get("fault_type"),
                "gt_layer": gt.get("layer"),
                "pred_layer": pr.get("layer"),
                "neighbors": (r.get("retrieval", {}) or {}).get("neighbors"),
            }
        else:
            rec = {
                "run_id": r.get("run_id"),
                "gt_ft": r.get("gt_ft"),
                "pred_ft": r.get("pred_ft"),
                "gt_layer": r.get("gt_layer"),
                "pred_layer": r.get("pred_layer"),
                "neighbors": None,
            }
        norm.append(rec)
    return norm


def _detect_system(norm: list) -> str:
    # The SN taxonomy shares no class with TT's TT-only set, so any TT-only label
    # uniquely identifies tt; SN is then identified by its own SN-only labels.
    gts = {rec["gt_ft"] for rec in norm if rec["gt_ft"] is not None}
    if gts & _TT_ONLY:
        return "tt"
    if gts & _SN_ONLY:
        return "sn"
    # Only shared classes present (e.g. cpu_pressure-only subset): default to tt.
    sys.stderr.write(
        "[score_unified] warning: could not disambiguate tt vs sn from gt labels; "
        "defaulting to tt (override with --system).\n"
    )
    return "tt"


def score_samples(norm: list, system: str) -> dict:
    """Compute all headline metrics + the three-state decomposition for one run."""
    n = len(norm)
    if n == 0:
        sys.exit("[score_unified] no samples found in result JSON.")

    missing = [rec for rec in norm if rec["gt_ft"] is None or rec["gt_layer"] is None]
    if missing:
        sys.exit(
            f"[score_unified] {len(missing)} samples are missing gt_ft/gt_layer — "
            "the result JSON does not match the expected nested/flat schema."
        )

    labels = FAULT_LABELS[system]
    # A prediction outside the fixed class set (e.g. a baseline's 'unknown') must
    # count as a miss, not crash sklearn — coerce None to a sentinel string.
    y_gt_ft = [rec["gt_ft"] for rec in norm]
    y_pred_ft = [rec["pred_ft"] if rec["pred_ft"] is not None else "__unpredicted__" for rec in norm]
    y_gt_ly = [rec["gt_layer"] for rec in norm]
    y_pred_ly = [rec["pred_layer"] if rec["pred_layer"] is not None else "__unpredicted__" for rec in norm]

    ft_correct = [g == p for g, p in zip(y_gt_ft, y_pred_ft)]
    ly_correct = [g == p for g, p in zip(y_gt_ly, y_pred_ly)]

    exact = sum(ft_correct)
    partial = sum(1 for fc, lc in zip(ft_correct, ly_correct) if (not fc) and lc)
    wrong = sum(1 for fc, lc in zip(ft_correct, ly_correct) if (not fc) and (not lc))
    triage_useful = exact + partial  # == sum(ft_correct OR (ft_wrong AND layer_correct))

    metrics = {
        "system": system,
        "num_samples": n,
        "fault_type_accuracy": exact / n,
        "layer_accuracy": sum(ly_correct) / n,
        "fault_type_macro_f1": f1_score(
            y_gt_ft, y_pred_ft, labels=labels, average="macro", zero_division=0
        ),
        "fault_type_macro_precision": precision_score(
            y_gt_ft, y_pred_ft, labels=labels, average="macro", zero_division=0
        ),
        "fault_type_macro_recall": recall_score(
            y_gt_ft, y_pred_ft, labels=labels, average="macro", zero_division=0
        ),
        "layer_macro_f1": f1_score(
            y_gt_ly, y_pred_ly, labels=LAYER_LABELS, average="macro", zero_division=0
        ),
        "triage": {
            "exact_diagnosis": exact / n,
            "partial_triage_utility": partial / n,
            "wrong_direction": wrong / n,
            "triage_useful_rate": triage_useful / n,
        },
        "counts": {"exact": exact, "partial": partial, "wrong": wrong, "n": n},
        "per_sample": [
            {
                "run_id": rec["run_id"],
                "gt_ft": rec["gt_ft"],
                "pred_ft": rec["pred_ft"],
                "gt_layer": rec["gt_layer"],
                "pred_layer": rec["pred_layer"],
                "ft_correct": fc,
                "layer_correct": lc,
            }
            for rec, fc, lc in zip(norm, ft_correct, ly_correct)
        ],
    }
    return metrics


def reuse_metrics(norm: list) -> dict | None:
    """Retrieval-reuse metrics from the as-run neighbors the LLM actually saw.

    Returns None for flat (baseline) schemas, which carry no neighbors.
    helps    = retrieval plurality correct AND final prediction correct.
    misleads = plurality wrong AND final prediction wrong AND prediction followed
               that wrong plurality (not merely wrong by coincidence).
    """
    have = [rec for rec in norm if rec.get("neighbors")]
    if not have:
        return None
    n = len(have)
    hit_ft = {k: 0 for k in HIT_KS}
    hit_ly = {k: 0 for k in HIT_KS}
    mrr_ft = mrr_ly = 0.0
    helps = misleads = 0

    for rec in have:
        nb = rec["neighbors"]
        gt_ft, gt_ly, pred_ft = rec["gt_ft"], rec["gt_layer"], rec["pred_ft"]
        fts = [x.get("fault_type") for x in nb]
        lys = [x.get("layer") for x in nb]

        for k in HIT_KS:
            if gt_ft in fts[:k]:
                hit_ft[k] += 1
            if gt_ly in lys[:k]:
                hit_ly[k] += 1

        rank_ft = next((i + 1 for i, f in enumerate(fts) if f == gt_ft), None)
        rank_ly = next((i + 1 for i, l in enumerate(lys) if l == gt_ly), None)
        mrr_ft += (1.0 / rank_ft) if rank_ft else 0.0
        mrr_ly += (1.0 / rank_ly) if rank_ly else 0.0

        plurality_ft = Counter(fts).most_common(1)[0][0] if fts else None
        if plurality_ft == gt_ft and pred_ft == gt_ft:
            helps += 1
        if plurality_ft != gt_ft and pred_ft != gt_ft and pred_ft == plurality_ft:
            misleads += 1

    return {
        "num_samples_with_neighbors": n,
        "hit_at_k_fault_type": {k: hit_ft[k] / n for k in HIT_KS},
        "hit_at_k_layer": {k: hit_ly[k] / n for k in HIT_KS},
        "mrr_fault_type": mrr_ft / n,
        "mrr_layer": mrr_ly / n,
        "retrieval_helps_rate": helps / n,
        "retrieval_misleads_rate": misleads / n,
        "retrieval_net_help": (helps - misleads) / n,
        "counts": {"helps": helps, "misleads": misleads, "n": n},
    }


def score_file(path: str, system: str | None = None) -> dict:
    """Convenience entry for stats.py: normalize + score one file, auto-detecting
    the system unless given. Returns the full score_samples dict (with per_sample)."""
    norm = _normalize(_load_results(path))
    sysd = system or _detect_system(norm)
    return score_samples(norm, sysd)


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _print_score(m: dict, src: str):
    t = m["triage"]
    print(f"\n{'='*68}")
    print(f"UNIFIED SCORE [{m['system']}]  ({m['num_samples']} samples)  {src}")
    print(f"{'='*68}")
    print(f"FT Macro-F1 (fixed 12-class) : {m['fault_type_macro_f1']:.3f}"
          f"   (macro-P {m['fault_type_macro_precision']:.3f} / macro-R {m['fault_type_macro_recall']:.3f})")
    print(f"FT Accuracy                  : {_pct(m['fault_type_accuracy'])}")
    print(f"Layer Accuracy               : {_pct(m['layer_accuracy'])}"
          f"   (layer Macro-F1 {m['layer_macro_f1']:.3f})")
    print(f"--- SRE triage utility (three-state) ---")
    print(f"  Exact diagnosis (FT ✓)         : {_pct(t['exact_diagnosis'])}   ({m['counts']['exact']}/{m['counts']['n']})")
    print(f"  Partial triage (FT ✗, Layer ✓) : {_pct(t['partial_triage_utility'])}   ({m['counts']['partial']}/{m['counts']['n']})")
    print(f"  Wrong direction (Layer ✗)      : {_pct(t['wrong_direction'])}   ({m['counts']['wrong']}/{m['counts']['n']})")
    print(f"  >> Triage-useful rate          : {_pct(t['triage_useful_rate'])}")


def _print_reuse(ru: dict | None, src: str):
    if ru is None:
        print(f"\n[reuse] {src}: flat schema — no retrieval.neighbors to score "
              "(reuse metrics apply to Ours/diagnose output only).")
        return
    print(f"\n{'='*68}")
    print(f"RETRIEVAL REUSE  ({ru['num_samples_with_neighbors']} samples)  {src}")
    print(f"{'='*68}")
    hf, hl = ru["hit_at_k_fault_type"], ru["hit_at_k_layer"]
    print(f"Same-FT    Hit@1/3/5 : {_pct(hf[1])} / {_pct(hf[3])} / {_pct(hf[5])}   (MRR {ru['mrr_fault_type']:.3f})")
    print(f"Same-layer Hit@1/3/5 : {_pct(hl[1])} / {_pct(hl[3])} / {_pct(hl[5])}   (MRR {ru['mrr_layer']:.3f})")
    print(f"Retrieval helps rate    : {_pct(ru['retrieval_helps_rate'])}   ({ru['counts']['helps']}/{ru['counts']['n']})")
    print(f"Retrieval misleads rate : {_pct(ru['retrieval_misleads_rate'])}   ({ru['counts']['misleads']}/{ru['counts']['n']})")
    print(f"Net help (helps−misleads): {_pct(ru['retrieval_net_help'])}")


def main():
    ap = argparse.ArgumentParser(
        description="Unified re-scorer for diagnose/baseline/ablation result JSONs."
    )
    ap.add_argument("cmd", choices=["score", "reuse"], help="which metrics to compute")
    ap.add_argument("result_json", help="result JSON (nested Ours or flat baseline)")
    ap.add_argument("--system", choices=["tt", "sn"], default=None,
                    help="force the class set; default auto-detects from gt labels")
    ap.add_argument("--output", default=None, help="write the metrics JSON here")

    # Default subcommand: bare 'score_unified.py results/x.json' → 'score'.
    argv = sys.argv[1:]
    if argv and argv[0] not in ("score", "reuse", "-h", "--help"):
        argv = ["score"] + argv
    args = ap.parse_args(argv)

    norm = _normalize(_load_results(args.result_json))
    system = args.system or _detect_system(norm)
    src = f"({Path(args.result_json).name})"

    if args.cmd == "score":
        out = score_samples(norm, system)
        _print_score(out, src)
    else:
        out = reuse_metrics(norm)
        _print_reuse(out, src)
        out = out if out is not None else {"note": "flat schema — no neighbors"}

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote metrics → {args.output}")


if __name__ == "__main__":
    main()
