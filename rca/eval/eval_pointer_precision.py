#!/usr/bin/env python
"""Pointer-precision evaluation for the baseline-relative evidence detector.

Scores detector evidence against **fault-injection ground truth** (meta.json),
never hand-written SOP rules:

  * **target-entity** (L2/L3/L4, target_service set): do the detector's
    trace / log flagged services include the injected ``target_service``?
    Reports hit@N and the target's rank. L1 is node-level (no target service) →
    N/A. Log is only scored on incidents where the log modality flagged
    anything (performance-only faults produce no log surge).
  * **metric resource-family** (resource faults only): are the top-N metric
    channels in the resource the injector physically acts on (stress→system_cpu,
    HikariCP→db_connections, JVM→jvm, kill→kube_pod/service_up, quota→
    kube_resourcequota)? This is injector physics, not a diagnostic heuristic.
    L2 (target metric goes dark) and http_error (non-resource) → N/A.
  * **metric discriminativeness** (all 12 types, zero prior): intra-class vs
    inter-class Jaccard of the top-K flagged channel sets — does the metric
    evidence carry stable, separable per-fault-type information?

Temporal precision is 100% by construction (the detector only reads the fault
window); the script asserts evidence timestamps fall in the window rather than
measuring it.

Run (>2 min, full test set re-reads telemetry):
    conda activate fullstack_rca
    python rca/eval/eval_pointer_precision.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.split import experiment_level_split
from fullstack_rca.detect import EvidenceDetector, parse_windows

# Resource the injector physically acts on → channel-name prefixes that must
# respond. This is injector physics (the fault directly drives these signals),
# NOT a diagnostic heuristic. Fault types whose evidence is not metric-resident
# map to None and are reported N/A.
RESOURCE_FAMILY = {
    "cpu_pressure": ("system_cpu_",),
    "disk_io_saturation": ("system_disk_",),
    "memory_pressure": ("system_memory_",),
    "jvm_memory_pressure": ("jvm_",),
    "database_connection_exhaustion": ("db_connections_",),
    "pod_kill": ("kube_pod_", "service_up_status"),
    "liveness_probe_failure": ("kube_pod_", "service_up_status"),
    "resource_quota_exhaustion": ("kube_resourcequota_",),
    "connection_reset": None,       # target metric goes dark — no channel "should" light up
    "network_latency": None,
    "packet_loss": None,
    "http_error_injection": None,   # injected HTTP 500 is not a resource metric
}


def _read_traces(exp_dir):
    """Flatten all trace spans (injecting processID→serviceName) and aggregate
    this experiment's service call edges (parent_service → child_service) from
    each trace's CHILD_OF references."""
    spans = []
    edges = set()
    for tf in glob.glob(os.path.join(exp_dir, "traces", "*.json")):
        try:
            data = json.load(open(tf)).get("data", [])
        except Exception:
            continue
        if not data:
            continue
        procs = data[0].get("processes", {})
        trace_spans = data[0].get("spans", [])
        sid2svc = {}
        for sp in trace_spans:
            pid = sp.get("processID")
            svc = procs.get(pid, {}).get("serviceName", "") if pid in procs else ""
            sp["_service_name"] = svc
            sid2svc[sp.get("spanID")] = svc
            spans.append(sp)
        for sp in trace_spans:
            csvc = sid2svc.get(sp.get("spanID"))
            for ref in sp.get("references", []):
                if ref.get("refType") == "CHILD_OF":
                    psvc = sid2svc.get(ref.get("spanID"))
                    if psvc and csvc and psvc != csvc:
                        edges.add((psvc, csvc))
    return spans, edges


def _top_services(scored, n):
    """Distinct services from a score-sorted evidence list, top-N, order kept."""
    seen = []
    for d in scored:
        svc = d.get("service")
        if svc and svc not in seen:
            seen.append(svc)
        if len(seen) >= n:
            break
    return seen


def _target_hit(scored, target, n):
    svcs = _top_services(scored, n)
    if target in svcs:
        return True, svcs.index(target) + 1
    return False, None


def _mean_jaccard(sets):
    pairs = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
    return float(np.mean(pairs)) if pairs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--num-samples", type=int, default=-1, help="-1 = all test experiments")
    p.add_argument("--top-n", type=int, default=8, help="evidence depth for hit@N (matches LLM view)")
    p.add_argument("--top-k-disc", type=int, default=5, help="top-K channels for discriminativeness")
    p.add_argument("--output", default="results/pointer_precision")
    args = p.parse_args()

    config = load_config(args.config)
    print("[1/3] Loading dataset (full index)...")
    dataset = create_dataset(config["dataset"])
    _, test_ids = experiment_level_split(dataset._samples_index, test_ratio=0.2, seed=42)

    exp_dirs = {}
    for info in dataset._samples_index:
        rid = info["run_id"]
        if rid in test_ids and rid not in exp_dirs:
            exp_dirs[rid] = info["exp_dir"]
    items = list(exp_dirs.items())
    if args.num_samples > 0:
        items = items[: args.num_samples]

    det = EvidenceDetector()
    print(f"[2/3] Scoring {len(items)} test experiments...")

    # per-fault-type tallies
    ft_layer = {}
    trace_hits = defaultdict(lambda: [0, 0])   # ft -> [hit, total]
    trace_ranks = defaultdict(list)
    trace1hop_hits = defaultdict(lambda: [0, 0])
    trace1hop_baseline = defaultdict(list)     # ft -> per-sample random hit prob
    log_hits = defaultdict(lambda: [0, 0])
    metricA_hits = defaultdict(lambda: [0, 0])
    disc_sets = defaultdict(list)              # ft -> list of top-K channel sets
    call_edges = set()
    pending_1hop = []                          # (ft, target, flagged_services)
    temporal_ok = True

    for rid, exp_dir in items:
        meta = json.loads((Path(exp_dir) / "meta.json").read_text())
        w = parse_windows(meta)
        ft, target = w.root_cause_type, w.target_service
        ft_layer[ft] = w.layer

        mser = dataset._load_experiment_metrics(exp_dir, rid)
        logs = dataset._load_experiment_logs(exp_dir, rid)
        spans, edges = _read_traces(exp_dir)
        call_edges |= edges

        mscored = det.score_metrics(mser, w)
        tscored = det.score_traces(det.normalize_spans(spans), w)
        lscored = det.score_logs(logs, w)

        # temporal: all flagged elements are inside the fault window by design.
        temporal_ok = temporal_ok and all(d.get("in_fault_window") for d in mscored + tscored + lscored)

        # target-entity (only when the fault has an injected target service)
        if target:
            hit, rank = _target_hit(tscored, target, args.top_n)
            trace_hits[ft][1] += 1
            if hit:
                trace_hits[ft][0] += 1
                trace_ranks[ft].append(rank)
            pending_1hop.append((ft, target, set(_top_services(tscored, args.top_n))))
            if lscored:  # log modality produced evidence for this incident
                lhit, _ = _target_hit(lscored, target, args.top_n)
                log_hits[ft][1] += 1
                log_hits[ft][0] += int(lhit)

        # metric resource-family (resource faults only)
        fam = RESOURCE_FAMILY.get(ft)
        if fam:
            top_names = [d["name"] for d in mscored[: args.top_n]]
            hit = any(nm.startswith(pref) or nm == pref
                      for nm in top_names for pref in fam)
            metricA_hits[ft][1] += 1
            metricA_hits[ft][0] += int(hit)

        # discriminativeness (all types)
        disc_sets[ft].append({d["name"] for d in mscored[: args.top_k_disc]})

    # inter-class Jaccard: mean over all cross-type pairs of channel sets
    inter_pairs = []
    fts = list(disc_sets)
    for i in range(len(fts)):
        for j in range(i + 1, len(fts)):
            for a in disc_sets[fts[i]]:
                for b in disc_sets[fts[j]]:
                    if a | b:
                        inter_pairs.append(len(a & b) / len(a | b))
    inter_jaccard = float(np.mean(inter_pairs)) if inter_pairs else None

    # target-or-direct-neighbour hit vs random baseline. Call graph is
    # aggregated from the test traces; the 1-hop neighbourhood = direct
    # caller/callee of the injected target (avoids the ≤2-hop blow-up where
    # the Gateway hub makes almost any service reachable).
    adj = defaultdict(set)
    for a, b in call_edges:
        adj[a].add(b)
        adj[b].add(a)
    n_nodes = len(adj)
    for ft, target, flagged in pending_1hop:
        nbr = {target} | adj.get(target, set())
        trace1hop_hits[ft][1] += 1
        if flagged & nbr:
            trace1hop_hits[ft][0] += 1
        if n_nodes:
            trace1hop_baseline[ft].append(len(nbr) / n_nodes)

    print("[3/3] Writing report...")
    _write(args, ft_layer, trace_hits, trace_ranks, trace1hop_hits,
           trace1hop_baseline, log_hits, metricA_hits, disc_sets,
           inter_jaccard, temporal_ok, args.top_n, args.top_k_disc)


def _rate(hits):
    h, t = hits
    return f"{h}/{t} ({100*h/t:.0f}%)" if t else "—"


def _write(args, ft_layer, trace_hits, trace_ranks, trace1hop_hits,
           trace1hop_baseline, log_hits, metricA_hits, disc_sets,
           inter_jaccard, temporal_ok, top_n, top_k):
    order = sorted(ft_layer, key=lambda f: (ft_layer[f], f))
    lines = ["# Pointer-Precision Report (fault-injection ground truth)", ""]
    lines.append(f"- evidence depth (hit@N): top-{top_n}  |  discriminativeness: top-{top_k} channels")
    lines.append(f"- **temporal precision: 100% by construction** "
                 f"({'verified' if temporal_ok else 'VIOLATION — flagged evidence outside fault window'})")
    lines.append(f"- inter-class metric Jaccard (lower = more separable): "
                 f"{inter_jaccard:.3f}" if inter_jaccard is not None else "- inter-class Jaccard: n/a")
    lines += ["", "| Layer | Fault type | trace target hit@N (rank) | trace target+1hop hit@N (rand) | log target hit@N | metric resource-family hit@N | intra Jaccard |",
              "|---|---|---|---|---|---|---|"]
    for ft in order:
        layer = ft_layer[ft]
        tr = trace_hits.get(ft)
        tr_cell = "N/A (node-level)" if tr is None else _rate(tr)
        if tr and trace_ranks[ft]:
            tr_cell += f" (r{np.mean(trace_ranks[ft]):.1f})"
        h1 = trace1hop_hits.get(ft)
        if h1 is None:
            h1_cell = "N/A (node-level)"
        else:
            base = np.mean(trace1hop_baseline[ft]) if trace1hop_baseline.get(ft) else 0.0
            h1_cell = f"{_rate(h1)} (rand {100*base:.0f}%)"
        lg = log_hits.get(ft)
        lg_cell = "N/A" if lg is None else _rate(lg)
        ma = metricA_hits.get(ft)
        ma_cell = "N/A" if RESOURCE_FAMILY.get(ft) is None else (_rate(ma) if ma else "—")
        intra = _mean_jaccard(disc_sets[ft])
        intra_cell = f"{intra:.3f}" if intra is not None else "—"
        lines.append(f"| {layer} | {ft} | {tr_cell} | {h1_cell} | {lg_cell} | {ma_cell} | {intra_cell} |")

    lines += ["", "## Reading guide",
              "- **trace target hit@N**: detector flagged services include the exact injected `target_service`. L1 = N/A (node-level, no target service).",
              "- **trace target+1hop hit@N (rand)**: flagged services include the target OR its direct caller/callee (call graph aggregated from traces). Compare to the random baseline `rand` — the Gateway-hub graph makes ≤2-hop trivial, so only **direct** neighbours count. hit-rate >> rand = real localisation to the fault's direct participants.",
              "- **metric resource-family hit@N**: top-N metric channels include the injected resource's family. N/A = fault not metric-resident (L2 target-dark, http_error).",
              "- **intra-class Jaccard** (higher = more stable) vs **inter-class** (lower = more separable)."]

    md = args.output + ".md"
    Path(md).parent.mkdir(parents=True, exist_ok=True)
    Path(md).write_text("\n".join(lines))
    payload = {
        "top_n": top_n, "top_k_disc": top_k, "temporal_ok": temporal_ok,
        "inter_class_jaccard": inter_jaccard,
        "per_fault_type": {
            ft: {
                "layer": ft_layer[ft],
                "trace_target": trace_hits.get(ft),
                "trace_avg_rank": float(np.mean(trace_ranks[ft])) if trace_ranks.get(ft) else None,
                "trace_target_or_1hop": trace1hop_hits.get(ft),
                "trace_1hop_random_baseline": float(np.mean(trace1hop_baseline[ft])) if trace1hop_baseline.get(ft) else None,
                "log_target": log_hits.get(ft),
                "metric_resource_family": metricA_hits.get(ft),
                "metric_resource_family_applicable": RESOURCE_FAMILY.get(ft) is not None,
                "intra_class_jaccard": _mean_jaccard(disc_sets[ft]),
            } for ft in order
        },
    }
    Path(args.output + ".json").write_text(json.dumps(payload, indent=2))
    print("\n".join(lines))
    print(f"\nSaved → {md}  +  {args.output}.json")


if __name__ == "__main__":
    main()
