#!/usr/bin/env python3
'English text removed.'
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

                                                                                    
                                                                                      
                                                                                
                                                                                 
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import yaml
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fullstack_rca.agent.llm_agent import LLMReasoningAgent
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.split import experiment_level_split
from fullstack_rca.rag.pipeline import RCAPipeline
from fullstack_rca.detect import EvidenceDetector, parse_windows


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end RCA Diagnosis Evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Source domain config (checkpoint + KB). Default: rca/config/dataset.yaml")
    parser.add_argument("--eval-mode", type=str, default="in_domain",
                        choices=["in_domain", "cross_domain"],
                        help="Evaluation mode (default: in_domain)")
    parser.add_argument("--query-config", type=str, default=None,
                        help="Target domain dataset config (cross_domain mode only)")
    parser.add_argument("--label-mapping", type=str, default=None,
                        help="Path to label_mapping.yaml (cross_domain mode)")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of test samples to diagnose (-1 = full test set)")
    parser.add_argument("--model", type=str, default="qwen2.5:14b",
                        help="LLM model name (default: qwen2.5:14b)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="LLM API base URL (default: OpenAI official)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="LLM API key (default: OPENAI_API_KEY env var)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of retrieved episodes (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full DiagnosisResult.to_text() for each sample")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sample evidence block (only OK/MISS one-liner)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (optional)")
    return parser.parse_args()


                           

def load_label_mapping(mapping_path: str, dataset_key: str = "tt_data") -> dict:
    """Load cross-domain label mapping from YAML.

    Returns:
        {source_root_cause: {layer, fault_type, confidence}} or empty dict.
    """
    if not mapping_path or not os.path.exists(mapping_path):
        return {}
    with open(mapping_path) as f:
        raw = yaml.safe_load(f)
    return raw.get(dataset_key, {})


def apply_label_mapping(gt_layer: str, gt_ft: str, mapping: dict):
    """Map a query-domain label to source-domain canonical labels.

    Returns:
        (mapped_layer, mapped_ft, confidence)
        confidence is 'exact', 'coarse', or 'unmapped'.
    """
    if not mapping:
        return gt_layer, gt_ft, "original"

    entry = mapping.get(gt_ft)
    if entry:
        return entry["layer"], entry["fault_type"], entry.get("confidence", "coarse")

    return gt_layer, gt_ft, "unmapped"


def _collect_experiments(samples_index, experiment_ids):
    """Group trace indices by experiment, in first-appearance order.

    Returns:
        List of (run_id, [trace_indices_in_dataset]) ordered by the first time
        each experiment appears in samples_index. Used so the symmetric query
        protocol (encode all traces, pool per experiment) sees each experiment
        exactly once, with stable sample numbering.
    """
    exp_key = "run_id" if "run_id" in samples_index[0] else "scenario_name"
    order = []
    grouped: dict = {}
    for i, info in enumerate(samples_index):
        eid = info[exp_key]
        if eid not in experiment_ids:
            continue
        if eid not in grouped:
            order.append(eid)
            grouped[eid] = []
        grouped[eid].append(i)
    return [(eid, grouped[eid]) for eid in order]


                                   

def get_raw_spans(dataset, sample_idx):
    'English text removed.'
    info = dataset._samples_index[sample_idx]
    trace_idx = info["trace_idx"]

    if hasattr(dataset, '_load_trace'):
        run_id = info.get("run_id")
                                     
        entry = dataset._trace_cache[run_id][trace_idx]
        trace_file = entry[1]
        if not isinstance(trace_file, list):
            try:
                with open(trace_file) as f:
                    raw = json.load(f)
                data = raw.get("data", [])
                if data:
                    spans = data[0].get("spans", [])
                    processes = data[0].get("processes", {})
                                   
                    for span in spans:
                        pid = span.get("processID")
                        if pid and pid in processes:
                            span["_service_name"] = processes[pid].get("serviceName", "")
                    return spans
            except Exception:
                pass
        _, spans = dataset._load_trace(run_id, trace_idx)
        return spans

    cache_key = info.get("scenario_name", info.get("experiment_id"))
    trace = dataset._trace_cache[cache_key][trace_idx]

    if isinstance(trace, dict) and "spans" in trace:
        return trace.get("spans", [])
    if isinstance(trace, dict) and "traceID" in trace:
        return trace.get("spans", [])

    return []


def get_raw_logs(dataset, sample_idx):
    'English text removed.'
    info = dataset._samples_index[sample_idx]

    if hasattr(dataset, '_collect_logs'):
                                     
        exp_dir = info.get("exp_dir", "")
        run_id = info.get("run_id", "")
        trace_id = info.get("trace_id", "")
        anchor_ts = info.get("inject_center_s", 0.0)
        try:
            return dataset._collect_logs(exp_dir, run_id, trace_id, anchor_ts)
        except Exception:
            return []

    return []


                     

def _gt_layer(r):
    return r["ground_truth"]["layer"]


def _gt_ft(r):
    return r["ground_truth"]["fault_type"]


def _pred_layer(r):
    return r["prediction"]["layer"]


def _pred_ft(r):
    return r["prediction"]["fault_type"]


def compute_macro_f1(results):
    'English text removed.'
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    all_types = set()
    for r in results:
        gt, pred = _gt_ft(r), _pred_ft(r)
        all_types.add(gt)
        all_types.add(pred)
        if gt == pred:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pred] += 1
    if not all_types:
        return 0.0
    f1s = []
    for t in all_types:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        rec = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def compute_metrics(results, eval_mode="in_domain"):
    'English text removed.'
    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return {}

    layer_correct = sum(1 for r in results if _gt_layer(r) == _pred_layer(r))
    ft_correct = sum(1 for r in results if _gt_ft(r) == _pred_ft(r))
    macro_f1 = compute_macro_f1(results)

    metrics = {
        "layer_accuracy": layer_correct / total,
        "fault_type_accuracy": ft_correct / total,
        "macro_f1": macro_f1,
    }

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ({total} samples, mode={eval_mode})")
    print(f"{'='*70}")
    print(f"Layer Accuracy:      {layer_correct:3d}/{total} = {layer_correct/total:.1%}")
    print(f"Fault Type Accuracy: {ft_correct:3d}/{total} = {ft_correct/total:.1%}")
    print(f"Macro F1:            {macro_f1:.4f}")

                                                         
    if eval_mode == "cross_domain":
        conf_counts = Counter(
            r["ground_truth"].get("mapping_confidence", "original") for r in results
        )
        print(f"\nLabel Mapping Coverage:")
        for conf_type in ["exact", "coarse", "unmapped", "original"]:
            c = conf_counts.get(conf_type, 0)
            if c > 0:
                print(f"  {conf_type}: {c}/{total} = {c/total:.1%}")

        mapped_results = [
            r for r in results
            if r["ground_truth"].get("mapping_confidence") in ("exact", "coarse")
        ]
        if mapped_results:
            m_total = len(mapped_results)
            m_layer = sum(1 for r in mapped_results if _gt_layer(r) == _pred_layer(r))
            m_ft = sum(1 for r in mapped_results if _gt_ft(r) == _pred_ft(r))
            print(f"\nMapped-only Accuracy (n={m_total}):")
            print(f"  Layer:     {m_layer}/{m_total} = {m_layer/m_total:.1%}")
            print(f"  FaultType: {m_ft}/{m_total} = {m_ft/m_total:.1%}")

                         
    layer_results = {}
    for r in results:
        gt = _gt_layer(r)
        if gt not in layer_results:
            layer_results[gt] = {"correct": 0, "total": 0}
        layer_results[gt]["total"] += 1
        if _gt_layer(r) == _pred_layer(r):
            layer_results[gt]["correct"] += 1

    print(f"\nPer-Layer Accuracy:")
    print(f"{'GT Layer':<12} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
    print("-" * 42)
    for layer in sorted(layer_results.keys()):
        lr = layer_results[layer]
        acc = lr["correct"] / lr["total"] if lr["total"] > 0 else 0
        print(f"{layer:<12} {lr['correct']:<10} {lr['total']:<8} {acc:<10.1%}")

    confusion = Counter()
    for r in results:
        confusion[(_gt_layer(r), _pred_layer(r))] += 1

    print(f"\nConfusion Matrix (GT → Pred):")
    for (gt, pred), count in sorted(confusion.items()):
        marker = "OK" if gt == pred else "MISS"
        print(f"  {gt:>8} → {pred:<8} : {count:3d} [{marker}]")

    return metrics


                                               

def _telemetry_evidence(context):
    """Snapshot of the baseline-relative scored evidence the LLM saw
    (fault-window vs the incident's own pre-fault baseline)."""
    t, l, m = context.trace_summary, context.logs_summary, context.metrics_summary
    return {
        "trace": {
            "num_spans": t.get("num_spans"),
            "fault_window_span_count": t.get("fault_window_span_count"),
            "error_rate": t.get("error_rate"),
            "services": t.get("services", []),
            "anomalies": t.get("anomalies", []),
        },
        "logs": {
            "num_logs": l.get("num_logs"),
            "anomaly_count": l.get("anomaly_count"),
            "anomalies": l.get("anomalies", []),
        },
        "metrics": {
            "num_metrics": m.get("num_metrics"),
            "active_metrics": m.get("active_metrics"),
            "anomaly_count": m.get("anomaly_count"),
            "anomalies": m.get("anomalies", []),
        },
    }


def _build_sample_record(
    *,
    run_id, sample, diagnosis, context,
    gt_layer, gt_ft, gt_layer_raw, gt_ft_raw, map_conf, eval_mode,
    neighbors, time_retrieve, time_predict, time_total,
):
    """Pack one sample's full diagnosis into the new nested JSON schema."""
                                                
                                                                    
                                                                              
                                                                               
                                                                              
    layer_clf = context.metadata.get("layer_classifier")
    rconf_meta = context.metadata.get("retrieval_confidence", {}) or {}
    ood_gate_triggered = bool(rconf_meta.get("ood_gate_triggered", False))
    llm_layer = diagnosis.root_cause_layer.value
    a_hybrid_bypass = False
    if layer_clf:
        lc_layer = layer_clf["layer"]
        if ood_gate_triggered:
            pred_layer = llm_layer
            layer_src = (
                f"LLM (OOD gate; classifier said {lc_layer} "
                f"{layer_clf['confidence']:.0%})"
            )
            a_hybrid_bypass = True
        else:
            pred_layer = lc_layer
            layer_src = f"classifier ({layer_clf['confidence']:.0%})"
    else:
        pred_layer = llm_layer
        layer_src = "LLM"

                                                                             
    gt_block = {"layer": gt_layer, "fault_type": gt_ft}
    if eval_mode == "cross_domain":
        gt_block["mapping_confidence"] = map_conf
        gt_block["original_layer"] = gt_layer_raw
        gt_block["original_fault_type"] = gt_ft_raw

                     
    rconf = context.metadata.get("retrieval_confidence", {}) or {}
    retrieval_block = {
        "confidence_level": rconf.get("level", "high"),
        "competing_types": rconf.get("competing_types", []),
        "neighbors": neighbors,
    }

                                                                                
    reasoning_block = {
        "summary": " ".join(diagnosis.reasoning_trace) if diagnosis.reasoning_trace else "",
        "reasoning_trace": list(diagnosis.reasoning_trace or []),
        "analysis_conclusion": diagnosis.analysis_conclusion or "",
        "supporting_evidence": list(diagnosis.supporting_evidence or []),
        "layer_supporting_evidence": list(diagnosis.layer_supporting_evidence or []),
        "layer_confounders_excluded": list(diagnosis.layer_confounders_excluded or []),
        "recommendations": list(diagnosis.recommendations or []),
    }

                                                                         
                                                                           
                                                                         
                                                                        
                                                             
    pred_ft = diagnosis.root_cause_type
    llm_ft_original = pred_ft
    b1_triggered = False
    retrieval_tau = context.metadata.get("retrieval_tau")
                                                                                 
                                                                             
                                                                            
    if retrieval_tau is not None and neighbors and not ood_gate_triggered:
        top1_sim = float(neighbors[0]["similarity"])
        plurality_counter = Counter(n["fault_type"] for n in neighbors)
        plurality_ft, plurality_count = plurality_counter.most_common(1)[0]
        if top1_sim >= retrieval_tau and plurality_ft != pred_ft:
            pred_ft = plurality_ft
            b1_triggered = True

    prediction_block = {
        "layer": pred_layer,
        "fault_type": pred_ft,
        "llm_layer": diagnosis.root_cause_layer.value,
        "layer_source": layer_src,
        "layer_classifier": layer_clf,
        "confidence": float(diagnosis.confidence),
        "affected_services": list(diagnosis.affected_services or []),
        "match": {
            "layer": gt_layer == pred_layer,
            "fault_type": gt_ft == pred_ft,
        },
    }
    if b1_triggered:
        prediction_block["llm_fault_type"] = llm_ft_original
        prediction_block["fault_type_source"] = (
            f"retrieval (sim={top1_sim:.3f}, tau={retrieval_tau:.4f}, "
            f"LLM disagreed)"
        )
        retrieval_block["top1_similarity"] = top1_sim
        retrieval_block["plurality_fault_type"] = plurality_ft
        retrieval_block["plurality_count"] = f"{plurality_count}/{len(neighbors)}"
        retrieval_block["tau_used"] = retrieval_tau

                                                                            
                                                                             
                               
    if ood_gate_triggered:
        retrieval_block["ood_gate_triggered"] = True
        retrieval_block["top1_similarity"] = rconf_meta.get("top1_similarity")
        retrieval_block["tau_used"] = rconf_meta.get("tau")
        if a_hybrid_bypass:
            prediction_block["lc_layer_overridden"] = layer_clf["layer"]
            prediction_block["lc_layer_confidence"] = float(layer_clf["confidence"])

    return {
        "run_id": run_id,
        "ground_truth": gt_block,
        "prediction": prediction_block,
        "retrieval": retrieval_block,
        "reasoning": reasoning_block,
        "telemetry_evidence": _telemetry_evidence(context),
        "timing": {
            "retrieve_seconds": float(time_retrieve),
            "predict_seconds": float(time_predict),
            "total_seconds": float(time_total),
        },
    }


                                         

def _fmt_kv(d, sep=", ", limit=None):
    """Render a flat dict as 'k=v' joined by sep. None/empty → '(none)'."""
    if not d:
        return "(none)"
    items = list(d.items())
    if limit is not None:
        items = items[:limit]
    return sep.join(f"{k}={v}" for k, v in items)


def _print_sample_evidence(record, *, quiet=False):
    """Print 4-line concise (quiet=True) or ~18-line evidence block per sample.

    Reads the new nested schema from _build_sample_record.
    """
    pred = record["prediction"]
    gt = record["ground_truth"]
    layer_match = "OK" if pred["match"]["layer"] else "MISS"
    ft_match = "OK" if pred["match"]["fault_type"] else "MISS"
    total_s = record["timing"]["total_seconds"]
    rconf = record["retrieval"]["confidence_level"]
    competing = record["retrieval"]["competing_types"]
    conf_tag = f"  [LOW-CONF: {', '.join(competing)}]" if rconf == "low" and competing else ""

    if quiet:
        print(f"  Prediction: {pred['layer']} / {pred['fault_type']}  "
              f"(layer via {pred['layer_source']})")
        print(f"  Layer: [{layer_match}]  FaultType: [{ft_match}]  "
              f"Time: {total_s:.1f}s{conf_tag}")
        return

    tev = record["telemetry_evidence"]
    trace = tev["trace"]
    logs = tev["logs"]
    metrics = tev["metrics"]

    print()
    print("  ▼ Telemetry seen by LLM")
    services = trace.get("services") or []
    services_n = len(services)
    print(f"    Trace:   {trace.get('num_spans', '?')} spans, "
          f"error_rate={trace.get('error_rate', '?')}, {services_n} services")
    slow = trace.get("slow_spans") or []
    if slow:
        print(f"             Slow: {', '.join(slow[:3])}")
    err_spans = trace.get("error_span_details") or []
    if err_spans:
        print(f"             Error spans: {', '.join(err_spans[:3])}")

    n_raw = logs.get("num_logs", "?")
    n_unique = logs.get("unique_event_count")
    sample_n = len(logs.get("log_messages_sample") or [])
    if n_unique is not None and n_unique != n_raw:
        log_hdr = f"{n_raw} raw → {n_unique} unique events; showing {sample_n} of {n_unique}"
    elif n_unique is not None:
        log_hdr = f"{n_raw} raw, {n_unique} unique; showing {sample_n}"
    else:
        log_hdr = f"{n_raw} collected; showing {sample_n}"
    print(f"    Logs:    {log_hdr}")
    kw_counts = logs.get("error_keyword_counts") or {}
    print(f"             Keywords: {_fmt_kv(kw_counts, limit=6)}")

    active_ch = metrics.get("active_metrics", 0)
    print(f"    Metrics: {metrics.get('anomalous_metric_count', 0)} anomalous "
          f"of {active_ch} active")
    for m in (metrics.get("anomalous_metrics") or [])[:3]:
        print(f"             {m.get('name', '?'):<28s} "
              f"mean={m.get('mean', '?')} std={m.get('std', '?')} "
              f"max={m.get('max', '?')}")
    cohort_mean = metrics.get("cohort_mean_std")
    if cohort_mean is not None:
        print(f"             high-variance rule: std > 2 × mean(stds of "
              f"{active_ch} active channels)  [i.e. std > 2 × {cohort_mean}]")

    print()
    print(f"  ▼ Top-3 retrieval ({rconf} confidence)")
    for i, n in enumerate(record["retrieval"]["neighbors"][:3]):
        marker = "  [BEST]" if i == 0 else ""
        print(f"    {n['layer']:<3s} {n['fault_type']:<32s} "
              f"sim={n['similarity']:.4f}{marker}")

    print()
    print("  ▼ LLM reasoning")
    summary = record["reasoning"]["summary"]
    if summary:
                                    
        words = summary.split()
        line = "    Synthesis:"
        for w in words:
            if len(line) + len(w) + 1 > 80:
                print(line)
                line = "               " + w
            else:
                line += " " + w
        if line.strip():
            print(line)
    rb = record["reasoning"]
    print(
        f"    Grounding: FT-support={len(rb.get('supporting_evidence') or [])}  "
        f"layer-support={len(rb.get('layer_supporting_evidence') or [])}  "
        f"layer-excluded={len(rb.get('layer_confounders_excluded') or [])}     "
        f"Confidence: {pred['confidence']:.0%}"
    )
    recs = record["reasoning"]["recommendations"]
    if recs:
        print(f"    Recommendations: {'; '.join(recs[:2])}")

    print()
    print("  ▼ Decision")
    print(f"    Prediction: {pred['layer']} / {pred['fault_type']}  "
          f"(layer via {pred['layer_source']})")
    print(f"    Layer: [{layer_match}]  FaultType: [{ft_match}]  "
          f"Time: {total_s:.1f}s{conf_tag}")


                             

def _md_escape(s):
    """Minimal escaping for table cells — replace newlines and pipes."""
    if s is None:
        return ""
    return str(s).replace("\n", " ").replace("|", "\\|")


def _render_summary_table(output_data):
    metrics = output_data.get("metrics") or {}
    n = output_data.get("num_evaluated", 0)
    lines = [
        "## Summary",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    if "layer_accuracy" in metrics:
        la = metrics["layer_accuracy"]
        lines.append(f"| Layer Accuracy | {la:.1%} ({int(round(la*n))}/{n}) |")
    if "fault_type_accuracy" in metrics:
        fa = metrics["fault_type_accuracy"]
        lines.append(f"| Fault Type Accuracy | {fa:.1%} ({int(round(fa*n))}/{n}) |")
    if "macro_f1" in metrics:
        lines.append(f"| Macro-F1 | {metrics['macro_f1']:.4f} |")
    lines.append(f"| Total samples | {n} |")
    if "llm_calls" in output_data:
        lines.append(f"| LLM calls | {output_data['llm_calls']} |")
    if "total_time_s" in output_data and n:
        avg = output_data["total_time_s"] / n
        lines.append(f"| Avg time / sample | {avg:.1f}s |")
    lines.append("")
    return lines


def _group_by_fault_type(results):
    """Group results by GT fault_type, sorted by group size desc then name."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = (_gt_layer(r), _gt_ft(r))
        groups[key].append(r)
                                                   
    return sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]),
    )


def _render_sample_section(record, idx_in_group):
    pred = record["prediction"]
    gt = record["ground_truth"]
    layer_ok = pred["match"]["layer"]
    ft_ok = pred["match"]["fault_type"]
    mark = "✓" if (layer_ok and ft_ok) else "❌"
    short_id = record["run_id"]

    lines = [
        f"#### {mark} Sample {idx_in_group} — `{short_id}`",
        f"**GT**: {gt['layer']} / `{gt['fault_type']}`  ",
        f"**Pred**: {pred['layer']} / `{pred['fault_type']}` "
        f"(confidence {pred['confidence']:.0%}, layer via {pred['layer_source']})",
        "",
    ]

                  
    tev = record["telemetry_evidence"]
    trace = tev["trace"]
    logs = tev["logs"]
    metrics = tev["metrics"]
    services = trace.get("services") or []

    lines.append("**What LLM saw** (fault-window vs own pre-fault baseline)")
    lines.append(
        f"- Trace: {trace.get('num_spans', '?')} spans, "
        f"error_rate={trace.get('error_rate', '?')}, {len(services)} services, "
        f"fault-window spans={trace.get('fault_window_span_count', '?')}"
        + (f" (`{', '.join(services[:5])}`{'...' if len(services) > 5 else ''})"
           if services else "")
    )
    for s in (trace.get("anomalies") or [])[:8]:
        lines.append(f"  - {_md_escape(s)}")
    lines.append(
        f"- Logs: {logs.get('num_logs', '?')} records, "
        f"{logs.get('anomaly_count', 0)} flagged templates (fault-window surge)"
    )
    for s in (logs.get("anomalies") or [])[:8]:
        lines.append(f"  - {_md_escape(s)}")
    lines.append(
        f"- Metrics: {metrics.get('anomaly_count', 0)} flagged of "
        f"{metrics.get('active_metrics', '?')} active (baseline-relative deviation)"
    )
    for s in (metrics.get("anomalies") or [])[:8]:
        lines.append(f"  - {_md_escape(s)}")
    lines.append("")

                   
    rconf = record["retrieval"]["confidence_level"]
    competing = record["retrieval"]["competing_types"]
    rconf_label = rconf
    if competing:
        rconf_label += f" — competing: {', '.join(competing)}"
    lines.append(f"**Top-3 retrieved cases** (confidence: {rconf_label})")
    lines.append("")
    lines.append("| Rank | Layer | Fault Type | Similarity |")
    lines.append("|------|-------|------------|------------|")
    for i, n in enumerate(record["retrieval"]["neighbors"][:3]):
        rank = f"{i + 1} ★" if i == 0 else str(i + 1)
        lines.append(
            f"| {rank} | {n['layer']} | `{_md_escape(n['fault_type'])}` "
            f"| {n['similarity']:.4f} |"
        )
    lines.append("")

               
    rblock = record["reasoning"]
    summary = rblock["summary"]
    if summary:
        lines.append(f"**Synthesis**: {_md_escape(summary)}")
        lines.append("")

                                                                                 
    layer_support = rblock.get("layer_supporting_evidence") or []
    layer_confound = rblock.get("layer_confounders_excluded") or []
    if layer_support or layer_confound:
        lines.append(f"**Layer grounding** for `{pred['layer']}`")
        lines.append("")
        lines.append("| Verdict | Modality | Element | Value | Rationale |")
        lines.append("|---|---|---|---|---|")
        for e in layer_support[:6]:
            mod = _md_escape(str(e.get("modality", "")))
            el = _md_escape(str(e.get("element", "?")))
            val_s = _md_escape(str(e.get("value", "")))
            why = _md_escape(str(e.get("why", "")))
            lines.append(f"| ✓ supports | {mod} | `{el}` | {val_s} | {why} |")
        for c in layer_confound[:6]:
            mod = _md_escape(str(c.get("modality", "")))
            el = _md_escape(str(c.get("element", "?")))
            sug = str(c.get("suggests_layer", "")).strip()
            verdict = f"✗ excluded (vs {_md_escape(sug)})" if sug else "✗ excluded"
            reason = _md_escape(str(c.get("reason", "")))
            lines.append(f"| {verdict} | {mod} | `{el}` | — | {reason} |")
        lines.append("")

                                                                               
    support = rblock.get("supporting_evidence") or []
    if support:
        lines.append(f"**Diagnosis grounding** for `{pred['fault_type']}`")
        lines.append("")
        lines.append("| Modality | Element | Value | Defining-signature rationale |")
        lines.append("|---|---|---|---|")
        for e in support[:6]:
            mod = _md_escape(str(e.get("modality", "")))
            el = _md_escape(str(e.get("element", "?")))
            val_s = _md_escape(str(e.get("value", "")))
            why = _md_escape(str(e.get("why", "")))
            lines.append(f"| {mod} | `{el}` | {val_s} | {why} |")
        lines.append("")

                                                                                
                            
    conclusion = rblock.get("analysis_conclusion") or ""
    if conclusion:
        lines.append(f"**Conclusion** (evidence-only): {_md_escape(conclusion)}")
        lines.append("")

    recs = record["reasoning"]["recommendations"]
    if recs:
        lines.append("**Recommendations**:")
        for r in recs[:5]:
            lines.append(f"- {r}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _generate_markdown_report(output_data, md_path):
    """Render output_data to a single Markdown file at md_path."""
    from datetime import datetime

    run_name = os.path.splitext(os.path.basename(md_path))[0]
    if run_name.endswith("_report"):
        run_name = run_name[: -len("_report")]
    model = output_data.get("model", "?")
    eval_mode = output_data.get("eval_mode", "?")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# FullStack-RCA Diagnosis Report",
        "",
        f"**Run**: `{run_name}` | **Model**: `{model}` | **Mode**: `{eval_mode}` "
        f"| **Generated**: {date_str}",
        "",
    ]
    lines.extend(_render_summary_table(output_data))

    results = output_data.get("results") or []
    if not results:
        lines.append("_No samples to report._")
    else:
        lines.append("## Per-Sample Diagnosis")
        lines.append("")
        lines.append(
            "> Samples are grouped by **GT fault type**. "
            "✓ both layer and fault_type correct.  "
            "❌ misprediction."
        )
        lines.append("")

        for (gt_layer, gt_ft), group in _group_by_fault_type(results):
            n = len(group)
            noun = "sample" if n == 1 else "samples"
            lines.append(f"### {gt_layer} / {gt_ft}  ({n} {noun})")
            lines.append("")
            for i, rec in enumerate(group, start=1):
                lines.extend(_render_sample_section(rec, i))

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved to: {md_path}")


                  

def main():
    args = parse_args()
    start_time = time.time()

    source_config = load_config(args.config)
    checkpoint_dir = source_config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    kb_dir = source_config.get("kb_dir", "knowledge_base")

    print("=== End-to-End RCA Diagnosis ===\n")
    print(f"Mode: {args.eval_mode}")
    print(f"Source config: {args.config or 'default'}")

                                    
    if args.eval_mode == "cross_domain":
        if not args.query_config:
            print("ERROR: --query-config is required for cross_domain mode.")
            sys.exit(1)

        query_config = load_config(args.query_config)
        source_path = source_config["dataset"]["path"]
        query_path = query_config["dataset"]["path"]

        if os.path.realpath(source_path) == os.path.realpath(query_path):
            print("WARNING: Source and query datasets point to the same path!")
            print(f"  Source: {source_path}")
            print(f"  Query:  {query_path}")
            print("  This is NOT a true cross-domain evaluation.")

        print(f"Query dataset: {query_config['dataset']['type']} ({query_path})")

        print("\n[1/4] Loading query dataset...")
        query_dataset = create_dataset(query_config["dataset"])
                                                                        
                                                                        
        all_exp_ids = {
            info.get("run_id", info.get("scenario_name", ""))
            for info in query_dataset._samples_index
        }
        test_experiments = _collect_experiments(query_dataset._samples_index, all_exp_ids)
        print(f"  Query experiments: {len(test_experiments)}, "
              f"total traces: {len(query_dataset)}")

        label_mapping = load_label_mapping(args.label_mapping)
    else:
        print(f"Dataset: {source_config['dataset']['type']} ({source_config['dataset']['path']})")

        print("\n[1/4] Loading dataset...")
        query_dataset = create_dataset(source_config["dataset"])

                                                                   
        train_exps, test_exps = experiment_level_split(
            query_dataset._samples_index, test_ratio=0.2, seed=args.seed,
        )
                                                                           
        test_experiments = _collect_experiments(query_dataset._samples_index, test_exps)
        total_test_traces = sum(len(idxs) for _, idxs in test_experiments)
        print(f"  Total: {len(query_dataset)}, Test experiments: {len(test_experiments)}, "
              f"Test traces (all per exp): {total_test_traces}")

        label_mapping = {}

                                  
    system = source_config["dataset"].get("params", {}).get("system", "tt")
    print("\n[2/4] Initializing pipeline...")
    pipeline = RCAPipeline(
        checkpoint_path=checkpoint_path,
        kb_dir=kb_dir,
        top_k=args.top_k,
        metric_files=getattr(query_dataset, "metric_files", None),
    )

                                                                       
    if pipeline.duration_stats:
        query_dataset.duration_stats = pipeline.duration_stats
        print(f"  Duration stats: mean={pipeline.duration_stats['mean']:.4f}, std={pipeline.duration_stats['std']:.4f}")

                                                                              
    pipeline.evidence_detector = EvidenceDetector()
    print("  Evidence detector: baseline-relative statistics (mean-shift z)")

    print("\n[3/4] Initializing LLM agent...")
    agent = LLMReasoningAgent(
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=0.1,
        system=system,
    )
    print(f"  Model: {args.model}")

                                
    if args.num_samples < 0:
        num_samples = len(test_experiments)
    else:
        num_samples = min(args.num_samples, len(test_experiments))
    num_test = len(test_experiments)
    leaked_run_ids: list[str] = []

    print(f"\n[4/4] Diagnosing {num_samples} test experiments...")
    results = []
    errors = 0
    total_llm_calls = 0

    for i in range(num_samples):
        run_id, trace_indices = test_experiments[i]

                                                                   
        samples_list = [query_dataset[idx] for idx in trace_indices]
        rep_sample = samples_list[0]                                                      
        gt_layer_raw = rep_sample.layer if rep_sample.layer else "Normal"
        gt_ft_raw = rep_sample.root_cause_type if rep_sample.root_cause_type else "normal"

        gt_layer, gt_ft, map_conf = apply_label_mapping(
            gt_layer_raw, gt_ft_raw, label_mapping,
        )

        print(f"\n--- Sample {i+1}/{num_samples} ---")
        print(f"  Run ID: {run_id}")
        print(f"  Traces in experiment: {len(trace_indices)}")
        if args.eval_mode == "cross_domain" and map_conf != "original":
            print(f"  Original GT: {gt_layer_raw} / {gt_ft_raw}")
            print(f"  Mapped GT:   {gt_layer} / {gt_ft} ({map_conf})")
        else:
            print(f"  Ground Truth: {gt_layer} / {gt_ft}")

        try:
            raw_spans_list = [get_raw_spans(query_dataset, idx) for idx in trace_indices]
            raw_log_texts_list = [get_raw_logs(query_dataset, idx) for idx in trace_indices]

                                                                                 
            info0 = query_dataset._samples_index[trace_indices[0]]
            exp_dir0, rid0 = info0["exp_dir"], info0["run_id"]
            with open(os.path.join(exp_dir0, "meta.json")) as _mf:
                windows = parse_windows(json.load(_mf))
            detector_input = {
                "metric_series": query_dataset._load_experiment_metrics(exp_dir0, rid0),
                "log_entries": query_dataset._load_experiment_logs(exp_dir0, rid0),
                "raw_spans": [sp for sl in raw_spans_list for sp in (sl or [])],
                "windows": windows,
            }

                                     
            t_retrieve_start = time.time()
                                                                              
                                                                                  
            context = pipeline.process_experiment(
                samples_list,
                detector_input=detector_input,
            )
            time_retrieve = time.time() - t_retrieve_start

                                          
                                                                                   
                                                                                
                                                                               
                                                                                
                                                        
            leaked_hits = []
            for ep in context.retrieved_episodes:
                full_ep = pipeline.knowledge_manager.get_episode(ep["episode_id"])
                if full_ep is not None and full_ep.experiment_id == run_id:
                    leaked_hits.append(ep["episode_id"])
            if leaked_hits:
                print(f"  [LEAK] query run_id={run_id} self-retrieved "
                      f"({len(leaked_hits)}/{len(context.retrieved_episodes)} hits); skipping")
                leaked_run_ids.append(run_id)
                continue

                                                        
            neighbors = []
            for ep in context.retrieved_episodes:
                neighbors.append({
                    "run_id": ep.get("trace_id", ep.get("id", "")),
                    "layer": ep["layer"],
                    "fault_type": ep["fault_type"],
                    "similarity": ep["similarity"],
                })

            print(f"  Retrieved {len(context.retrieved_episodes)} episodes:")
            for ep in context.retrieved_episodes[:5]:
                print(f"    {ep['layer']:7s} {ep['fault_type']:25s} sim={ep['similarity']:.4f}")

                                          
            print(f"  Running LLM diagnosis ({args.model})...")
            t_predict_start = time.time()
            diagnosis = agent.diagnose(context)
            time_predict = time.time() - t_predict_start
            time_total = time_retrieve + time_predict

                                                                          
            total_llm_calls += 1

            record = _build_sample_record(
                run_id=run_id,
                sample=rep_sample,
                diagnosis=diagnosis,
                context=context,
                gt_layer=gt_layer,
                gt_ft=gt_ft,
                gt_layer_raw=gt_layer_raw,
                gt_ft_raw=gt_ft_raw,
                map_conf=map_conf,
                eval_mode=args.eval_mode,
                neighbors=neighbors,
                time_retrieve=time_retrieve,
                time_predict=time_predict,
                time_total=time_total,
            )
            results.append(record)
            _print_sample_evidence(record, quiet=args.quiet)

            if args.verbose:
                print(diagnosis.to_text())

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            continue

                                              
    if leaked_run_ids:
        print(f"\n[GUARD] {len(leaked_run_ids)}/{num_samples} samples skipped "
              f"due to self-retrieval (split/KB inconsistency):")
        for rid in leaked_run_ids:
            print(f"  - {rid}")
    else:
        print(f"\n[GUARD] {num_samples}/{num_samples} samples passed "
              f"self-retrieval check (no train/test leak detected).")

                         
    metrics = compute_metrics(results, eval_mode=args.eval_mode)

    total_time = time.time() - start_time
    avg_time = (
        sum(r["timing"]["total_seconds"] for r in results) / len(results)
        if results else 0
    )
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Avg time per sample: {avg_time:.1f}s")
    print(f"Errors: {errors}/{num_samples}")

                                                                
    output_path = args.output
    if not output_path:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        output_path = os.path.join(
            results_dir,
            f"fullstack_rca_{args.eval_mode}_{num_samples}samples.json",
        )

    output_data = {
        "method": "FullStack-RCA",
        "paper": "Ours",
        "model": args.model,
        "temperature": 0.1,
        "seed": args.seed,
        "top_k": args.top_k,
        "embed_dim": 512,
        "num_total_traces": len(query_dataset),
        "num_test_experiments": num_test,
        "num_evaluated": len(results),
        "errors": errors,
        "metrics": metrics,
        "total_time_s": total_time,
        "llm_calls": total_llm_calls,
        "eval_mode": args.eval_mode,
        "results": results,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    md_path = (
        output_path[:-5] + "_report.md"
        if output_path.endswith(".json")
        else output_path + "_report.md"
    )
    _generate_markdown_report(output_data, md_path)


if __name__ == "__main__":
    main()
