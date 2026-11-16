#!/usr/bin/env python
"""Localization + causal-chain evaluation for the LLM diagnosis.

Grades what the LLM contributes ON TOP of retrieval/classifier: a root-cause
**service** and an ordered cause->propagation->symptom **chain**. Scored against
fault-injection ground truth (meta.json), and against the real service call graph
the LLM never saw.

Two ground-truth regimes (every fault type is gradable on one of them):

  * **service-culprit faults** (L2, L4, L3 pod_kill/liveness_probe — target_service
    set): the LLM's localized_service / candidate_roots vs the injected
    target_service -> Hit@1 / Hit@3. Compared against two references the LLM must
    beat: (a) the naive "pick the highest-latency trace service" detector baseline
    (the gateway-hub trap), and (b) a random 1/n pointer.
  * **resource faults** (L1 cpu/disk/memory + L3 resource_quota — no culprit
    service): the chain root's signal must name the injected resource family
    (RESOURCE_FAMILY) AND the LLM must NOT false-pinpoint a single ts-* service
    (scope correctness).

  * **all 12 types — propagation-edge fidelity / fabrication rate**: every
    service->service hop in the chain must be a real edge in this incident's call
    graph (built here from the raw spans; the LLM reasoned WITHOUT it). Fabricated
    hops = the LLM inventing a propagation path that does not exist.

A second result JSON (--baseline-result) is graded the same way (its
affected_services as the ranked list) to quantify the contribution vs the prior
hallucinated-affected_services behavior.

Run (>2 min — re-reads every test experiment's traces):
    conda activate fullstack_rca
    python rca/eval/eval_localization.py --result results/<new_run>.json \
        [--baseline-result results/<old_run>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.split import experiment_level_split
from fullstack_rca.detect import EvidenceDetector, parse_windows

# Reuse the injector-physics resource map and the per-experiment call-graph
# builder from the pointer-precision eval (single source of truth).
from eval_pointer_precision import RESOURCE_FAMILY, _read_traces, _top_services


def _norm(s):
    return (s or "").strip().lower()


def _ranked_roots(pred):
    """Ranked candidate root services from a prediction block. Works on the new
    schema (localized_service + candidate_roots) and the old one (affected_services)."""
    ranked = []
    loc = pred.get("localized_service")
    if loc:
        ranked.append(loc)
    for c in (pred.get("candidate_roots") or []):
        if c not in ranked:
            ranked.append(c)
    if not ranked:  # old-schema fallback
        ranked = list(pred.get("affected_services") or [])
    return ranked


def _hit(ranked, target, n):
    target = _norm(target)
    return any(_norm(s) == target for s in ranked[:n])


def _chain_service_edges(chain):
    """Consecutive (node_i, node_{i+1}) pairs in the chain where BOTH nodes look
    like real services (ts-*). Non-service nodes (resource/node/namespace) are
    skipped — they cannot appear in a service call graph."""
    nodes = [str(step.get("node", "")) for step in (chain or [])]
    pairs = []
    for a, b in zip(nodes, nodes[1:]):
        if a.startswith("ts-") and b.startswith("ts-"):
            pairs.append((a, b))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, help="diagnosis result JSON to grade")
    p.add_argument("--baseline-result", default=None,
                   help="optional prior result JSON for contribution comparison")
    p.add_argument("--config", default=None)
    p.add_argument("--output", default="results/localization")
    args = p.parse_args()

    config = load_config(args.config)
    print("[1/3] Loading dataset index for run_id -> exp_dir map ...")
    dataset = create_dataset(config["dataset"])
    _, test_ids = experiment_level_split(dataset._samples_index, test_ratio=0.2, seed=42)
    exp_dirs = {}
    for info in dataset._samples_index:
        rid = info["run_id"]
        if rid in test_ids and rid not in exp_dirs:
            exp_dirs[rid] = info["exp_dir"]

    det = EvidenceDetector()

    # Per-experiment ground truth + structural facts, computed once and shared by
    # both the main result and the baseline result.
    print(f"[2/3] Building ground truth + call graphs for {len(exp_dirs)} experiments ...")
    gt = {}  # rid -> {ft, layer, target, edges(undirected adj), naive_top_service}
    for rid, exp_dir in exp_dirs.items():
        meta = json.loads((Path(exp_dir) / "meta.json").read_text())
        w = parse_windows(meta)
        spans, edges = _read_traces(exp_dir)
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        tscored = det.score_traces(det.normalize_spans(spans), w)
        naive_top = _top_services(tscored, 1)
        gt[rid] = {
            "ft": w.root_cause_type, "layer": w.layer,
            "target": w.target_service, "adj": adj,
            "naive_top": naive_top[0] if naive_top else None,
        }

    print("[3/3] Grading ...")
    main_stats = _grade(args.result, gt)
    base_stats = _grade(args.baseline_result, gt) if args.baseline_result else None
    _write(args, main_stats, base_stats)


def _grade(result_path, gt):
    """Grade one result JSON against the shared ground truth `gt`."""
    data = json.load(open(result_path))
    records = data.get("results", data) if isinstance(data, dict) else data

    # service-culprit tallies: ft -> [hit, total]
    svc_hit1 = defaultdict(lambda: [0, 0])
    svc_hit3 = defaultdict(lambda: [0, 0])
    naive_hit1 = defaultdict(lambda: [0, 0])
    rand_hit1 = defaultdict(list)          # ft -> per-sample 1/n
    # resource tallies
    res_family = defaultdict(lambda: [0, 0])
    res_scope = defaultdict(lambda: [0, 0])
    # fabrication (all)
    fab_edges = defaultdict(lambda: [0, 0])  # ft -> [fabricated, total chain edges]
    ft_layer = {}

    for r in records:
        rid = r.get("run_id")
        g = gt.get(rid)
        if g is None:
            continue
        ft, layer, target, adj = g["ft"], g["layer"], g["target"], g["adj"]
        ft_layer[ft] = layer
        pred = r.get("prediction", {})
        ranked = _ranked_roots(pred)
        chain = pred.get("root_cause_chain") or []

        if target:  # service-culprit regime
            svc_hit1[ft][1] += 1
            svc_hit1[ft][0] += int(_hit(ranked, target, 1))
            svc_hit3[ft][1] += 1
            svc_hit3[ft][0] += int(_hit(ranked, target, 3))
            naive_hit1[ft][1] += 1
            naive_hit1[ft][0] += int(_norm(g["naive_top"]) == _norm(target))
            n_nodes = max(len(adj), 1)
            rand_hit1[ft].append(1.0 / n_nodes)
        else:        # resource regime (L1 + quota)
            fam = RESOURCE_FAMILY.get(ft)
            if fam:
                root_sig = " ".join(
                    str(s.get("signal", "")) for s in chain if s.get("role") == "root"
                ) or " ".join(str(s.get("signal", "")) for s in chain)
                res_family[ft][1] += 1
                res_family[ft][0] += int(any(pref in root_sig for pref in fam))
            # scope: must NOT pin a single ts-* microservice as the culprit
            loc = _norm(pred.get("localized_service"))
            res_scope[ft][1] += 1
            res_scope[ft][0] += int(not loc.startswith("ts-"))

        # fabrication (all types): chain service-hops must be real call edges
        for a, b in _chain_service_edges(chain):
            fab_edges[ft][1] += 1
            real = b in adj.get(a, set()) or a in adj.get(b, set())
            fab_edges[ft][0] += int(not real)

    return {
        "ft_layer": ft_layer, "svc_hit1": svc_hit1, "svc_hit3": svc_hit3,
        "naive_hit1": naive_hit1, "rand_hit1": rand_hit1,
        "res_family": res_family, "res_scope": res_scope, "fab_edges": fab_edges,
        "path": result_path,
    }


def _rate(hits):
    h, t = hits
    return f"{h}/{t} ({100*h/t:.0f}%)" if t else "—"


def _agg(tally):
    h = sum(v[0] for v in tally.values())
    t = sum(v[1] for v in tally.values())
    return h, t


def _write(args, S, B):
    ft_layer = S["ft_layer"]
    order = sorted(ft_layer, key=lambda f: (ft_layer[f], f))
    L = ["# Localization & Causal-Chain Report (LLM contribution)", ""]
    L.append(f"- result: `{S['path']}`"
             + (f"  vs baseline: `{B['path']}`" if B else ""))
    L.append("")
    L += ["## Service-culprit faults — root-service localization (vs injected target_service)",
          "",
          "| Layer | Fault type | LLM Hit@1 | LLM Hit@3 | naive top-latency Hit@1 | random Hit@1 |",
          "|---|---|---|---|---|---|"]
    for ft in order:
        if not S["svc_hit1"].get(ft):
            continue
        rand = np.mean(S["rand_hit1"][ft]) if S["rand_hit1"].get(ft) else 0.0
        L.append(f"| {ft_layer[ft]} | {ft} | {_rate(S['svc_hit1'][ft])} "
                 f"| {_rate(S['svc_hit3'][ft])} | {_rate(S['naive_hit1'][ft])} "
                 f"| {100*rand:.0f}% |")
    h1, t1 = _agg(S["svc_hit1"]); h3, t3 = _agg(S["svc_hit3"])
    hn, tn = _agg(S["naive_hit1"])
    L.append(f"| **all** | **service-culprit** | **{_rate([h1, t1])}** "
             f"| **{_rate([h3, t3])}** | **{_rate([hn, tn])}** | — |")

    L += ["", "## Resource faults — root-signal family + scope correctness", "",
          "| Layer | Fault type | root-signal family hit | scope-ok (no false service pin) |",
          "|---|---|---|---|"]
    for ft in order:
        if not (S["res_family"].get(ft) or S["res_scope"].get(ft)):
            continue
        fam_cell = _rate(S["res_family"][ft]) if S["res_family"].get(ft) else "N/A"
        L.append(f"| {ft_layer[ft]} | {ft} | {fam_cell} | {_rate(S['res_scope'][ft])} |")

    L += ["", "## Propagation-edge fidelity (all types) — fabricated chain hops", "",
          "| Layer | Fault type | fabricated / total chain edges |", "|---|---|---|"]
    for ft in order:
        if not S["fab_edges"].get(ft):
            continue
        L.append(f"| {ft_layer[ft]} | {ft} | {_rate(S['fab_edges'][ft])} |")
    fh, ftot = _agg(S["fab_edges"])
    fab_pct = f"{100*fh/ftot:.0f}%" if ftot else "—"
    L.append(f"| **all** | — | **{fh}/{ftot} fabricated ({fab_pct})** |")

    if B:
        bh1, bt1 = _agg(B["svc_hit1"]); bh3, bt3 = _agg(B["svc_hit3"])
        L += ["", "## Contribution vs baseline (service-culprit localization)", "",
              "| Result | Hit@1 | Hit@3 |", "|---|---|---|",
              f"| this run | {_rate([h1, t1])} | {_rate([h3, t3])} |",
              f"| baseline `{os.path.basename(B['path'])}` | {_rate([bh1, bt1])} | {_rate([bh3, bt3])} |"]

    L += ["", "## Reading guide",
          "- **LLM Hit@1/@3 >> naive top-latency Hit@1**: the LLM looked past the gateway-hub (highest-latency symptom) to the real culprit — the localization contribution retrieval/classifier cannot make.",
          "- **root-signal family hit**: for node/namespace faults the chain root names the injected resource (disk->system_disk_*, etc.).",
          "- **scope-ok**: the LLM correctly did NOT pin a single microservice for an infra-wide fault.",
          "- **fabricated chain edges**: chain service->service hops absent from the real call graph = invented propagation. Lower is better; the LLM never saw the graph, so this is an honest faithfulness check."]

    md = args.output + ".md"
    Path(md).parent.mkdir(parents=True, exist_ok=True)
    Path(md).write_text("\n".join(L))
    payload = {
        "result": S["path"], "baseline": B["path"] if B else None,
        "service_culprit": {
            "hit@1": {"hit": h1, "total": t1},
            "hit@3": {"hit": h3, "total": t3},
            "naive_top_latency_hit@1": {"hit": hn, "total": tn},
        },
        "fabricated_edges": {"fabricated": fh, "total": ftot},
        "per_fault_type": {
            ft: {
                "layer": ft_layer[ft],
                "svc_hit@1": S["svc_hit1"].get(ft),
                "svc_hit@3": S["svc_hit3"].get(ft),
                "naive_top_hit@1": S["naive_hit1"].get(ft),
                "resource_family_hit": S["res_family"].get(ft),
                "scope_ok": S["res_scope"].get(ft),
                "fabricated_edges": S["fab_edges"].get(ft),
            } for ft in order
        },
    }
    Path(args.output + ".json").write_text(json.dumps(payload, indent=2))
    print("\n".join(L))
    print(f"\nSaved → {md}  +  {args.output}.json")


if __name__ == "__main__":
    main()
