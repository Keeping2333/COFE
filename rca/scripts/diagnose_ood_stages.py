#!/usr/bin/env python3
"""OOD-service 5-stage diagnostic protocol.

For each of the 24 round2_ood_service samples, walk the diagnosis pipeline
and emit per-stage Pass/Fail to attribute the dominant failure mode.

Stages:
  1. Raw telemetry  — does GT-layer-specific signal exist in raw logs/metrics?
  2. Encoder        — does OOD embedding land near correct-layer KB region?  (TODO)
  3. Retrieval+LC   — do LC + top-K agree with GT layer?                      (TODO)
  4. Summary        — does prompt's telemetry summary preserve L-signals?     (TODO)
  5. LLM            — does LLM pick GT layer given the prompt it received?    (TODO)

Stage 1 only for now (--stage 1). Later stages dispatched by --stage 2..5.

Usage:
    cd <artifact-root>
    conda activate fullstack_rca
    python rca/scripts/diagnose_ood_stages.py --stage 1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

                                                                          
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT / "rca" / "src"))

from fullstack_rca.data.exp_runs_dataset import _METRIC_FILES              


DEFAULT_OOD_DIR = PROJ_ROOT / "train-ticket" / "exp_runs" / "round2_ood_service"


                                                                            
                                         
                                                                             
                                                                           
                                                            
                                                                        
                                               
 
                                                                        
                                                                         
                                                                    
                                                                     
                                      

LAYER_METRICS: Dict[str, set] = {
    "L1": {
                                                                       
        "system_cpu_utilization", "system_cpu_by_mode", "system_cpu_load_1m",
        "system_cpu_load_5m", "system_memory_usage", "system_memory_utilization",
        "system_disk_io_bytes", "system_disk_operations", "system_disk_io_time",
        "system_disk_weighted_io_time", "system_filesystem_usage",
    },
    "L2": {
                                                                              
        "network_io_bytes", "network_packets", "network_errors",
        "network_dropped", "network_connections_by_state",
        "coredns_request_count", "coredns_request_duration_p99",
        "coredns_responses_total", "coredns_response_nxdomain",
        "coredns_response_servfail", "coredns_panics_total",
        "coredns_cache_hits",
    },
    "L3": {
                                                                                   
        "kube_pod_phase", "kube_pod_ready", "kube_pod_restarts_total",
        "kube_pod_waiting_reason", "kube_deployment_status_replicas",
        "kube_deployment_status_replicas_available",
        "kube_deployment_status_replicas_unavailable",
        "kube_resourcequota_hard", "kube_resourcequota_used",
        "service_up_status", "scrape_targets_health",
    },
    "L4": {
                                                                              
        "http_request_rate", "http_request_latency_p99",
        "http_request_latency_p95", "http_request_latency_p50",
        "http_request_latency_avg", "http_client_request_rate",
        "http_client_latency_avg",
        "jvm_memory_used", "jvm_memory_limit", "jvm_memory_committed",
        "jvm_gc_duration", "jvm_gc_count", "jvm_thread_count",
        "jvm_class_count", "jvm_cpu_utilization", "jvm_cpu_time",
        "db_connections_usage", "db_connections_max",
        "db_connections_idle_min", "db_connections_pending",
        "db_connections_wait_time", "db_connections_use_time",
        "db_connections_create_time",
    },
}

LAYER_LOG_KEYWORDS: Dict[str, set] = {
    "L1": {
                                           
        "oom", "oomkilled", "killed", "out of memory", "memory pressure",
    },
    "L2": {
                                                      
        "dns", "nxdomain", "servfail",
        "name resolution", "unknown host", "no route to host",
        "network unreachable", "connection refused", "connection reset",
        "host unreachable",
    },
    "L3": {
                                            
        "oomkilled", "evicted", "imagepullbackoff", "errimagepull",
        "image pull", "back-off", "backoff",
        "probe failed", "liveness", "readiness",
        "quota exceeded", "forbidden", "crashloopbackoff",
        "containercreating", "podinitializing",
    },
    "L4": {
                                                    
        "exhausted", "pool exhausted", "hikari", "hikaripool",
        "connection pool", "request timed out",
        "transactionexception", "cannotcreatetransaction",
        "outofmemoryerror", "java.lang.outofmemoryerror",
        "garbage collection",
    },
}

ALL_LOG_KEYWORDS: set = set()
for kws in LAYER_LOG_KEYWORDS.values():
    ALL_LOG_KEYWORDS.update(kws)


def _parse_iso(s: str) -> float:
    """Parse ISO-8601 UTC string to unix timestamp (seconds)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


                                                                        
                                         
                                                                        

def _load_metric_series(metric_dir: Path, fname: str) -> np.ndarray:
    """Load one Prometheus JSON channel and aggregate multi-series by ts mean.

    Mirrors exp_runs_dataset._load_experiment_metrics so the std we compute
    here matches what the pipeline would see at inference time.
    """
    p = metric_dir / fname
    if not p.exists():
        return np.array([], dtype=np.float32)
    try:
        with open(p) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return np.array([], dtype=np.float32)
    result = raw.get("data", {}).get("result", [])
    all_ts, all_vals = [], []
    for r in result:
        for ts_val, val_str in r.get("values", []):
            try:
                all_ts.append(float(ts_val))
                all_vals.append(float(val_str))
            except (ValueError, TypeError):
                continue
    if not all_ts:
        return np.array([], dtype=np.float32)
    ts_arr = np.array(all_ts, dtype=np.float64)
    val_arr = np.array(all_vals, dtype=np.float32)
    unique_ts = np.unique(ts_arr)
    agg = np.array(
        [val_arr[ts_arr == t].mean() for t in unique_ts],
        dtype=np.float32,
    )
    return agg


def _load_metric_with_ts(metric_dir: Path, fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Like _load_metric_series but returns (timestamps, values) before aggregation.

    Aggregation by ts (mean across services) is still applied so we get one
    value per unique timestamp.
    """
    p = metric_dir / fname
    if not p.exists():
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
    try:
        with open(p) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
    result = raw.get("data", {}).get("result", [])
    all_ts, all_vals = [], []
    for r in result:
        for ts_val, val_str in r.get("values", []):
            try:
                v = float(val_str)
            except (ValueError, TypeError):
                continue
            try:
                t = float(ts_val)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(v):
                continue
            all_ts.append(t)
            all_vals.append(v)
    if not all_ts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)
    ts_arr = np.array(all_ts, dtype=np.float64)
    val_arr = np.array(all_vals, dtype=np.float32)
    unique_ts = np.unique(ts_arr)
    agg = np.array(
        [val_arr[ts_arr == t].mean() for t in unique_ts],
        dtype=np.float32,
    )
    return unique_ts, agg


def _channel_anomalous_during_injection(
    metric_dir: Path, fname: str, inj_start: float, inj_end: float,
) -> Dict[str, Any]:
    """Decide whether one metric channel exhibits an injection-window anomaly.

    Criterion (no arbitrary constants — uses the channel's own baseline distribution):
      anomalous_during == True  iff
          max(during) > max(baseline)  AND  max(during) > 0
      where baseline = points outside [inj_start, inj_end]
            during   = points inside  [inj_start, inj_end]
      For event-counter channels with baseline_max == 0 (e.g. nxdomain,
      network_errors), any non-zero during-window observation triggers True.
      For continuous channels, max(during) > max(baseline) means the
      injection produced a value the channel never reached before.

    Returns a dict with the verdict and supporting stats.
    """
    ts, vals = _load_metric_with_ts(metric_dir, fname)
    if ts.size == 0:
        return {"present": False, "reason": "no_data"}
    during_mask = (ts >= inj_start) & (ts <= inj_end)
    baseline_mask = ~during_mask
    d = vals[during_mask]
    b = vals[baseline_mask]
    if d.size == 0:
        return {"present": False, "reason": "no_during_pts"}
    dmax = float(d.max())
    bmax = float(b.max()) if b.size > 0 else 0.0
    dmean = float(d.mean())
    bmean = float(b.mean()) if b.size > 0 else 0.0
    anomalous = (dmax > bmax) and (dmax > 0)
    return {
        "present": anomalous,
        "during_max": dmax,
        "baseline_max": bmax,
        "during_mean": dmean,
        "baseline_mean": bmean,
        "n_during": int(d.size),
        "n_baseline": int(b.size),
    }


def _count_log_keywords(log_file: Path) -> Dict[str, int]:
    """Count case-insensitive keyword hits across LogRecord bodies.

    Mirrors pipeline._summarize_logs keyword scan, but expanded to the
    union of L1+L2+L3+L4 keywords so we can attribute per-layer signals.
    """
    counts: Dict[str, int] = {kw: 0 for kw in ALL_LOG_KEYWORDS}
    if not log_file.exists():
        return {}
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    batch = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for rl in batch.get("resourceLogs", []) or []:
                    for sl in rl.get("scopeLogs", []) or []:
                        for lr in sl.get("logRecords", []) or []:
                            body = ((lr.get("body") or {}).get("stringValue") or "").lower()
                            if not body:
                                continue
                            for kw in ALL_LOG_KEYWORDS:
                                if kw in body:
                                    counts[kw] += 1
    except OSError:
        pass
    return {kw: c for kw, c in counts.items() if c > 0}


def stage1_for_sample(exp_dir: Path) -> Dict[str, Any]:
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)
    anomaly = meta.get("anomaly", {}) or {}
    gt_layer = anomaly.get("layer", "?")
                                                               
                                                
    gt_ft = anomaly.get("root_cause_type") or anomaly.get("fault_type") or "?"

    ts = meta.get("timestamps", {}) or {}
    inj_start = _parse_iso(ts["inject_start"])
    inj_end = _parse_iso(ts["inject_end"])

    layer_metric_set = LAYER_METRICS.get(gt_layer, set())
    layer_keyword_set = LAYER_LOG_KEYWORDS.get(gt_layer, set())

                                                                             
    metric_dir = exp_dir / "metrics"
    metric_hits: Dict[str, Dict[str, Any]] = {}
    metric_checked: Dict[str, Dict[str, Any]] = {}
    for fname in _METRIC_FILES:
        stem = Path(fname).stem
        if stem not in layer_metric_set:
            continue
        verdict = _channel_anomalous_during_injection(
            metric_dir, fname, inj_start, inj_end,
        )
        metric_checked[stem] = verdict
        if verdict.get("present"):
            metric_hits[stem] = verdict

                                                                              
    keyword_counts = _count_log_keywords(exp_dir / "logs" / "logs.jsonl")
    keyword_hits = {kw: c for kw, c in keyword_counts.items() if kw in layer_keyword_set}

    s1_pass = bool(metric_hits) or bool(keyword_hits)

    return {
        "id": exp_dir.name,
        "gt_layer": gt_layer,
        "gt_ft": gt_ft,
        "target_service": anomaly.get("target_service") or anomaly.get("targetService") or "?",
        "injection_window": {"start": inj_start, "end": inj_end, "duration_s": inj_end - inj_start},
        "S1": "P" if s1_pass else "F",
        "S1_evidence": {
            "metric_hits": metric_hits,
            "keyword_hits": keyword_hits,
        },
        "S1_diagnostics": {
            "metric_channels_checked": list(metric_checked.keys()),
            "metric_channels_with_data": [
                k for k, v in metric_checked.items() if "during_max" in v
            ],
            "all_log_keyword_counts": keyword_counts,
        },
    }


                                                                        
                                                             
                                                                        

                                                                             
                                                                          
                                                                         
                                                                            
                                                                          
                                             
 
                                                                       
                                                                      
                                                                            
TRACKED_KEYWORDS_ROOT: Dict[str, str] = {
                      
    "OutOfMemoryError": "jvm_memory_pressure",
    "java.lang.OutOfMemoryError": "jvm_memory_pressure",
    "Java heap space": "jvm_memory_pressure",
    "GC overhead limit": "jvm_memory_pressure",
                     
    "HikariPool": "db_connection_exhaustion",
    "Connection is not available": "db_connection_exhaustion",
                      
    "UnknownHostException": "dns_failure",
    "Name or service not known": "dns_failure",
    "no such host": "dns_failure",
    "nodename nor servname": "dns_failure",
                          
    "Connection timed out": "network",
    "Connection refused": "network",
    "No route to host": "network",
    "Network is unreachable": "network",
                                                                         
                                      
    "Liveness probe failed": "liveness_probe_failure",
    "Readiness probe failed": "liveness_probe_failure",
    "Health check failed": "liveness_probe_failure",
}

TRACKED_KEYWORDS_CASCADE: Dict[str, str] = {
                           
    "TransactionException": "cascade_transaction",
    "CannotCreateTransactionException": "cascade_transaction",
    "JpaSystemException": "cascade_transaction",
                      
    "request timed out after 30000ms": "cascade_pool_timeout",
    "request timed out": "cascade_pool_timeout",
    "Read timed out": "cascade_pool_timeout",
                       
    "500 Internal Server Error": "cascade_5xx",
    "Bad Gateway": "cascade_5xx",
    "Service Unavailable": "cascade_5xx",
}

ALL_TRACKED = {**TRACKED_KEYWORDS_ROOT, **TRACKED_KEYWORDS_CASCADE}


def _first_occurrence_timeline(
    log_file: Path, inj_start: float, inj_end: float,
) -> Dict[str, Any]:
    """For each tracked keyword (case-sensitive), find the FIRST
    LogRecord whose body contains it, after inject_start.

    Returns:
        {
          "first_after_inject": {kw: rel_seconds_after_inject_start},
          "first_overall": {kw: absolute_ts_seconds},  # may be before inject
          "n_total_records": int,
          "n_records_scanned_post_inject": int,
        }
    """
    first_after: Dict[str, float] = {}
    first_overall: Dict[str, float] = {}
    n_total = 0
    n_post = 0
    if not log_file.exists():
        return {"first_after_inject": {}, "first_overall": {},
                "n_total_records": 0, "n_records_scanned_post_inject": 0}
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    batch = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for rl in batch.get("resourceLogs", []) or []:
                    for sl in rl.get("scopeLogs", []) or []:
                        for lr in sl.get("logRecords", []) or []:
                            n_total += 1
                            ts_ns_raw = (
                                lr.get("timeUnixNano")
                                or lr.get("observedTimeUnixNano")
                            )
                            try:
                                ts_s = (
                                    int(ts_ns_raw) / 1e9 if ts_ns_raw else 0.0
                                )
                            except (TypeError, ValueError):
                                ts_s = 0.0
                            body = (
                                (lr.get("body") or {}).get("stringValue")
                                or ""
                            )
                            if not body:
                                continue
                            if ts_s >= inj_start:
                                n_post += 1
                            for kw in ALL_TRACKED:
                                if kw in body:
                                    if (kw not in first_overall
                                            or ts_s < first_overall[kw]):
                                        first_overall[kw] = ts_s
                                    if ts_s >= inj_start:
                                        if (kw not in first_after
                                                or ts_s < first_after[kw]):
                                            first_after[kw] = ts_s
    except OSError:
        pass

    rel = {kw: (ts - inj_start) for kw, ts in first_after.items()}
    return {
        "first_after_inject": rel,
        "first_overall": first_overall,
        "n_total_records": n_total,
        "n_records_scanned_post_inject": n_post,
        "inject_window_seconds": inj_end - inj_start,
    }


def log_timing_for_sample(exp_dir: Path) -> Dict[str, Any]:
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)
    anomaly = meta.get("anomaly", {}) or {}
    gt_layer = anomaly.get("layer", "?")
    gt_ft = anomaly.get("root_cause_type") or anomaly.get("fault_type") or "?"
    ts = meta.get("timestamps", {}) or {}
    inj_start = _parse_iso(ts["inject_start"])
    inj_end = _parse_iso(ts["inject_end"])

    timeline = _first_occurrence_timeline(
        exp_dir / "logs" / "logs.jsonl", inj_start, inj_end,
    )

                                                                           
    rel = timeline["first_after_inject"]
    root_hits = {
        kw: rel[kw] for kw in TRACKED_KEYWORDS_ROOT if kw in rel
    }
    cascade_hits = {
        kw: rel[kw] for kw in TRACKED_KEYWORDS_CASCADE if kw in rel
    }
    earliest_root = (
        min(root_hits.items(), key=lambda x: x[1]) if root_hits else None
    )
    earliest_cascade = (
        min(cascade_hits.items(), key=lambda x: x[1])
        if cascade_hits else None
    )

                                                                
    inferred_ft = None
    if earliest_root is not None:
        inferred_ft = TRACKED_KEYWORDS_ROOT[earliest_root[0]]

    return {
        "id": exp_dir.name,
        "gt_layer": gt_layer,
        "gt_ft": gt_ft,
        "inject_window_seconds": timeline["inject_window_seconds"],
        "n_records_post_inject": timeline["n_records_scanned_post_inject"],
        "earliest_root_keyword": (
            {"keyword": earliest_root[0],
             "rel_seconds": earliest_root[1],
             "inferred_ft": inferred_ft}
            if earliest_root else None
        ),
        "earliest_cascade_keyword": (
            {"keyword": earliest_cascade[0],
             "rel_seconds": earliest_cascade[1]}
            if earliest_cascade else None
        ),
        "all_root_first_seconds": root_hits,
        "all_cascade_first_seconds": cascade_hits,
    }


def run_log_timing(ood_dir: Path, output_path: Path) -> None:
    samples = sorted(
        d for d in ood_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    print("Sub-experiment 2: log first-occurrence timing")
    print(f"OOD dir: {ood_dir}")
    print(f"Found {len(samples)} experiments\n")

    print(
        f"{'#':>2} {'sample':<58} {'gt_ft':<30}  "
        f"earliest_root(s)              earliest_cascade(s)"
    )
    print("-" * 160)
    results = []
    for idx, exp_dir in enumerate(samples, 1):
        r = log_timing_for_sample(exp_dir)
        results.append(r)
        er = r["earliest_root_keyword"]
        ec = r["earliest_cascade_keyword"]
        er_s = (
            f"{er['keyword'][:25]}({er['rel_seconds']:+.0f}s)→{er['inferred_ft'][:8]}"
            if er else "—"
        )
        ec_s = (
            f"{ec['keyword'][:25]}({ec['rel_seconds']:+.0f}s)"
            if ec else "—"
        )
        print(
            f"{idx:>2} {r['id'][:58]:<58} {r['gt_ft'][:30]:<30}  "
            f"{er_s:<35} {ec_s}"
        )

                                                                    
                                 
    by_gt_ft_inferred: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        er = r["earliest_root_keyword"]
        inferred = er["inferred_ft"] if er else "no_root_keyword"
        by_gt_ft_inferred[r["gt_ft"]][inferred] += 1

    print("\n" + "=" * 80)
    print(
        "Per-GT-FT: distribution of inferred root FT "
        "(based on earliest-root keyword after inject_start)"
    )
    print("=" * 80)
    print(f"{'gt_ft':<33}{'n':>3}  inferred (count)")
    for ft in sorted(by_gt_ft_inferred):
        counter = by_gt_ft_inferred[ft]
        n = sum(counter.values())
        parts = [
            f"{inf}={c}" for inf, c in counter.most_common()
        ]
        print(f"{ft:<33}{n:>3}  {', '.join(parts)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "log_timing",
            "ood_dir": str(ood_dir),
            "n_samples": len(samples),
            "tracked_keywords_root": TRACKED_KEYWORDS_ROOT,
            "tracked_keywords_cascade": TRACKED_KEYWORDS_CASCADE,
            "samples": results,
            "rule_notes": {
                "matching": "case-SENSITIVE substring on LogRecord.body.stringValue",
                "window": "first occurrence at or after meta.timestamps.inject_start",
                "rel_seconds": "first_occurrence_unix - inject_start_unix",
                "root_vs_cascade": (
                    "ROOT keywords = direct fault signature "
                    "(OutOfMemoryError, HikariPool, UnknownHost, "
                    "Liveness probe). CASCADE keywords = downstream "
                    "symptoms (TransactionException, request timed out, "
                    "500). Earliest ROOT after inject_start is the "
                    "inferred fault type."
                ),
            },
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


                                   
from collections import Counter              


                                                                        
                                                     
                                                                        

                                                                      
                                                                  
                                                                         
SLOPE_CHANNELS = [
    "jvm_memory_used",
    "jvm_memory_limit",
    "jvm_gc_count",
    "jvm_gc_duration",
    "jvm_cpu_utilization",
    "jvm_thread_count",
    "db_connections_usage",
    "db_connections_pending",
    "db_connections_wait_time",
    "http_request_rate",
    "http_request_latency_p99",
]


def _load_target_service_series(
    metric_path: Path, target_service: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, values) for the series whose
    metric.service_name == target_service. If multiple matching series
    exist (e.g. pod restart), we take all values sorted by timestamp.
    """
    if not metric_path.exists():
        return np.array([]), np.array([])
    try:
        with open(metric_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return np.array([]), np.array([])
    pts = []
    for r in raw.get("data", {}).get("result", []) or []:
        if (r.get("metric") or {}).get("service_name") != target_service:
            continue
        for tv, vs in r.get("values", []) or []:
            try:
                pts.append((float(tv), float(vs)))
            except (ValueError, TypeError):
                continue
    if not pts:
        return np.array([]), np.array([])
    pts.sort()
    return (
        np.array([p[0] for p in pts], dtype=np.float64),
        np.array([p[1] for p in pts], dtype=np.float64),
    )


def _slope_features(
    ts_arr: np.ndarray, val_arr: np.ndarray,
    inj_start: float, inj_end: float,
) -> Dict[str, Any]:
    if ts_arr.size == 0:
        return {"available": False, "reason": "no_series"}
    pre_mask = ts_arr < inj_start
    dur_mask = (ts_arr >= inj_start) & (ts_arr <= inj_end)
    pre = val_arr[pre_mask]
    dur = val_arr[dur_mask]
    if dur.size < 2 or pre.size < 2:
        return {
            "available": False,
            "reason": f"insufficient_points(pre={int(pre.size)},dur={int(dur.size)})",
        }
                              
    s_pre = float(np.polyfit(ts_arr[pre_mask], pre, 1)[0])
    s_dur = float(np.polyfit(ts_arr[dur_mask], dur, 1)[0])
                                                 
    v0, v1 = float(dur[0]), float(dur[-1])
    rel_growth = (v1 - v0) / max(abs(v0), 1e-9) if v0 != 0 else (
        float("inf") if v1 != 0 else 0.0
    )
    return {
        "available": True,
        "pre_mean": float(pre.mean()),
        "pre_slope_per_sec": s_pre,
        "during_mean": float(dur.mean()),
        "during_slope_per_sec": s_dur,
        "during_value_first": v0,
        "during_value_last": v1,
        "during_relative_growth": (
            None if rel_growth == float("inf") else rel_growth
        ),
        "during_relative_growth_pct": (
            None if rel_growth == float("inf") else 100.0 * rel_growth
        ),
        "during_minus_pre_mean": float(dur.mean() - pre.mean()),
    }


def metric_slopes_for_sample(exp_dir: Path) -> Dict[str, Any]:
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)
    anomaly = meta.get("anomaly", {}) or {}
    gt_layer = anomaly.get("layer", "?")
    gt_ft = anomaly.get("root_cause_type") or anomaly.get("fault_type") or "?"
    target_svc = anomaly.get("target_service") or anomaly.get("targetService") or "?"
    ts = meta.get("timestamps", {}) or {}
    inj_start = _parse_iso(ts["inject_start"])
    inj_end = _parse_iso(ts["inject_end"])

    per_channel: Dict[str, Any] = {}
    for ch in SLOPE_CHANNELS:
        arr_t, arr_v = _load_target_service_series(
            exp_dir / "metrics" / (ch + ".json"), target_svc,
        )
        per_channel[ch] = _slope_features(arr_t, arr_v, inj_start, inj_end)

    return {
        "id": exp_dir.name,
        "gt_layer": gt_layer,
        "gt_ft": gt_ft,
        "target_service": target_svc,
        "per_channel": per_channel,
    }


def run_metric_slopes(ood_dir: Path, output_path: Path) -> None:
    samples = sorted(
        d for d in ood_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    print("Sub-experiment 3: target-service metric slopes")
    print(f"OOD dir: {ood_dir}")
    print(f"Found {len(samples)} experiments\n")

                                                                               
    print(
        f"{'#':>2} {'sample':<58} {'gt_ft':<30}  "
        f"heap_growth%  gc_count_d/p   db_usage_d  http_rate_d"
    )
    print("-" * 165)
    results = []
    for idx, exp_dir in enumerate(samples, 1):
        r = metric_slopes_for_sample(exp_dir)
        results.append(r)
        pc = r["per_channel"]

        def fmt_growth(ch):
            f = pc.get(ch, {})
            if not f.get("available"):
                return "n/a"
            g = f.get("during_relative_growth_pct")
            return f"{g:+.1f}%" if g is not None else "∞"

        def fmt_mean(ch):
            f = pc.get(ch, {})
            if not f.get("available"):
                return "n/a"
            return f"{f.get('during_mean', 0):.3f}"

        heap = fmt_growth("jvm_memory_used")
        gc = f"{fmt_mean('jvm_gc_count')}"
        db = fmt_mean("db_connections_usage")
        http_rate = fmt_mean("http_request_rate")
        print(
            f"{idx:>2} {r['id'][:58]:<58} {r['gt_ft'][:30]:<30}  "
            f"{heap:<13} {gc:<14} {db:<11} {http_rate}"
        )

                                                                            
    print("\n" + "=" * 80)
    print(
        "Per-FT median target-service heap growth + DB / GC during injection"
    )
    print("=" * 80)
    print(
        f"{'fault_type':<33}{'n':>3}  "
        f"{'heap_growth%':>15}{'gc_count_dur':>14}"
        f"{'db_usage_dur':>14}{'http_rate_dur':>14}"
    )
    by_ft: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_ft[r["gt_ft"]].append(r["per_channel"])
    for ft in sorted(by_ft):
        rows = by_ft[ft]
        n = len(rows)
        def med(getter):
            xs = sorted(x for x in (getter(row) for row in rows) if x is not None)
            return xs[len(xs)//2] if xs else None
        hg = med(lambda pc: pc.get("jvm_memory_used", {}).get("during_relative_growth_pct"))
        gc = med(lambda pc: pc.get("jvm_gc_count", {}).get("during_mean"))
        db = med(lambda pc: pc.get("db_connections_usage", {}).get("during_mean"))
        ht = med(lambda pc: pc.get("http_request_rate", {}).get("during_mean"))
        hg_s = f"{hg:+.1f}%" if hg is not None else "—"
        gc_s = f"{gc:.4f}" if gc is not None else "—"
        db_s = f"{db:.3f}" if db is not None else "—"
        ht_s = f"{ht:.4f}" if ht is not None else "—"
        print(f"{ft:<33}{n:>3}  {hg_s:>15}{gc_s:>14}{db_s:>14}{ht_s:>14}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "metric_slopes",
            "ood_dir": str(ood_dir),
            "n_samples": len(samples),
            "channels": SLOPE_CHANNELS,
            "samples": results,
            "rule_notes": {
                "series_selection": (
                    "Filtered by metric.service_name == anomaly.target_service "
                    "(cluster-mean aggregation hides per-pod fault signal)"
                ),
                "slope": "numpy.polyfit degree-1 over (timestamp_s, value)",
                "relative_growth": (
                    "(value_at_inj_end - value_at_inj_start) / "
                    "max(|value_at_inj_start|, 1e-9)"
                ),
            },
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


                                                                        
                                                            
                                                                        

def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Index-based percentile on a pre-sorted list (no numpy interp)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * pct)))
    return sorted_vals[k]


def _extract_trace_features(
    traces_dir: Path,
    inj_start: float,
    inj_end: float,
) -> Dict[str, Any]:
    """Walk traces/*.json (Jaeger query schema), bucket spans per service /
    per window (during-injection vs baseline), and compute 5xx rate,
    error-tag rate, latency p50/p95/p99.

    Jaeger schema per file:
        {data: [{traceID, spans:[{processID, startTime(us), duration(us),
                                  tags:[{key,type,value}]}], processes: {pid: {serviceName}}}]}

    Returns:
        {
          "services": {svc: {"during": {n, n_5xx, n_error, p50, p95, p99},
                              "baseline": {n, n_5xx, n_error, p50, p95, p99}}},
          "n_trace_files": int,
        }
    """
                                                                       
                                                                       
                                                                            
                                        
     
                                                                            
                                                                           
                                                                       
                                                                      
                                      
    spans_buckets: Dict[str, Dict[str, List[Tuple[Any, bool, int]]]] = defaultdict(
        lambda: {"pre": [], "during": [], "post": []}
    )
    inj_start_us = inj_start * 1_000_000.0
    inj_end_us = inj_end * 1_000_000.0
    n_files = 0
    if not traces_dir.is_dir():
        return {"services": {}, "n_trace_files": 0}
    for trace_file in traces_dir.glob("*.json"):
        n_files += 1
        try:
            with open(trace_file) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for trace in d.get("data", []) or []:
            processes = trace.get("processes", {}) or {}
            svc_lookup = {
                pid: (p or {}).get("serviceName", "?")
                for pid, p in processes.items()
            }
            for span in trace.get("spans", []) or []:
                pid = span.get("processID", "")
                svc = svc_lookup.get(pid, "?")
                start_us = float(span.get("startTime", 0) or 0)
                duration_us = int(span.get("duration", 0) or 0)
                tags_list = span.get("tags", []) or []
                status_code = None
                error_flag = False
                for t in tags_list:
                    k = t.get("key", "")
                    v = t.get("value")
                    if k == "http.response.status_code":
                        status_code = v
                    elif k == "otel.status_code":
                        if str(v).upper() == "ERROR":
                            error_flag = True
                if start_us < inj_start_us:
                    window = "pre"
                elif start_us <= inj_end_us:
                    window = "during"
                else:
                    window = "post"
                spans_buckets[svc][window].append(
                    (status_code, error_flag, duration_us)
                )

                                                                             
    services: Dict[str, Dict[str, Any]] = {}
    for svc, windows in spans_buckets.items():
        services[svc] = {}
        for win, spans in windows.items():
            n = len(spans)
            n_5xx = sum(
                1 for s in spans
                if s[0] is not None and str(s[0]).startswith("5")
            )
            n_error = sum(1 for s in spans if s[1])
            durations = sorted(s[2] for s in spans)
            services[svc][win] = {
                "n_spans": n,
                "n_5xx": n_5xx,
                "n_error_tag": n_error,
                "rate_5xx": (n_5xx / n) if n else 0.0,
                "rate_error_tag": (n_error / n) if n else 0.0,
                "latency_p50_us": _percentile(durations, 0.50),
                "latency_p95_us": _percentile(durations, 0.95),
                "latency_p99_us": _percentile(durations, 0.99),
            }
    return {"services": services, "n_trace_files": n_files}


def trace_features_for_sample(exp_dir: Path) -> Dict[str, Any]:
    with open(exp_dir / "meta.json") as f:
        meta = json.load(f)
    anomaly = meta.get("anomaly", {}) or {}
    gt_layer = anomaly.get("layer", "?")
    gt_ft = anomaly.get("root_cause_type") or anomaly.get("fault_type") or "?"
    target_svc = anomaly.get("target_service") or anomaly.get("targetService") or "?"
    ts = meta.get("timestamps", {}) or {}
    inj_start = _parse_iso(ts["inject_start"])
    inj_end = _parse_iso(ts["inject_end"])

    features = _extract_trace_features(
        exp_dir / "traces", inj_start, inj_end,
    )

    target_block = features["services"].get(target_svc)

                                                                            
                                                                           
                                                                         
    global_agg: Dict[str, Dict[str, Any]] = {}
    for win in ("pre", "during", "post"):
        n_spans = 0
        n_5xx = 0
        n_error_tag = 0
        for svc, blocks in features["services"].items():
            b = blocks.get(win, {})
            n_spans += b.get("n_spans", 0)
            n_5xx += b.get("n_5xx", 0)
            n_error_tag += b.get("n_error_tag", 0)
        global_agg[win] = {
            "n_spans": n_spans,
            "n_5xx": n_5xx,
            "n_error_tag": n_error_tag,
            "rate_5xx_over_all_spans": (n_5xx / n_spans) if n_spans else 0.0,
            "rate_error_tag_over_all_spans": (n_error_tag / n_spans) if n_spans else 0.0,
        }

                                                                          
                                                                
    def _during_pre_ratio(svc_blocks: Dict[str, Any]) -> float:
        d = svc_blocks.get("during", {})
        p = svc_blocks.get("pre", {})
        rd = d.get("rate_5xx", 0)
        rp = p.get("rate_5xx", 0)
                                                                             
                                      
        if rd == 0:
            return 0.0
        if rp == 0:
            return 1e9
        return rd / rp

    top_5xx_during = sorted(
        [
            (svc, blocks.get("during", {}).get("n_5xx", 0),
             blocks.get("during", {}).get("rate_5xx", 0))
            for svc, blocks in features["services"].items()
        ],
        key=lambda x: -x[1],
    )[:5]
    top_5xx_elevation = sorted(
        [
            (svc, _during_pre_ratio(blocks),
             blocks.get("during", {}).get("rate_5xx", 0),
             blocks.get("pre", {}).get("rate_5xx", 0))
            for svc, blocks in features["services"].items()
        ],
        key=lambda x: -x[1],
    )[:5]

    return {
        "id": exp_dir.name,
        "gt_layer": gt_layer,
        "gt_ft": gt_ft,
        "target_service": target_svc,
        "n_trace_files": features["n_trace_files"],
        "n_services_observed": len(features["services"]),
        "global_windows": global_agg,
        "target_service_features": target_block,
        "top5_5xx_services_by_count_during": [
            {"service": s, "n_5xx": n, "rate_5xx": r}
            for s, n, r in top_5xx_during if n > 0
        ],
        "top5_5xx_services_by_elevation": [
            {"service": s, "during_pre_ratio": (None if r >= 1e9 else r),
             "rate_5xx_during": d, "rate_5xx_pre": p}
            for s, r, d, p in top_5xx_elevation if r > 0
        ],
        "all_services": features["services"],
    }


def run_trace_features(ood_dir: Path, output_path: Path) -> None:
    samples = sorted(
        d for d in ood_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    print(f"Sub-experiment: per-service trace feature extraction")
    print(f"OOD dir: {ood_dir}")
    print(f"Found {len(samples)} experiments\n")

    print(
        f"{'#':>2} {'sample':<58} {'gt_ft':<32}  "
        f"glob_5xx_d/p%  top_elev_svc(during%/pre%)"
    )
    print("-" * 165)
    results = []
    for idx, exp_dir in enumerate(samples, 1):
        r = trace_features_for_sample(exp_dir)
        results.append(r)
        g = r["global_windows"]
        r5 = (
            f"{100*g['during']['rate_5xx_over_all_spans']:.1f}/"
            f"{100*g['pre']['rate_5xx_over_all_spans']:.1f}"
        )
        elev = r["top5_5xx_services_by_elevation"]
        if elev:
            e0 = elev[0]
            elev_str = (
                f"{e0['service']}({100*e0['rate_5xx_during']:.0f}%/"
                f"{100*e0['rate_5xx_pre']:.0f}%)"
            )
        else:
            elev_str = "—"
        print(
            f"{idx:>2} {r['id'][:58]:<58} {r['gt_ft'][:32]:<32}  "
            f"{r5:<14} {elev_str}"
        )

                                                                          
    by_ft_global: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_ft_global[r["gt_ft"]].append(r["global_windows"])

    print("\n" + "=" * 80)
    print("Per-FT median GLOBAL error rates: during vs PRE-injection baseline")
    print("=" * 80)
    print(
        f"{'fault_type':<35}{'n':>3}  "
        f"{'5xx_d':>8}{'5xx_pre':>8}{'5xx_ratio':>11}  "
        f"{'err_d':>8}{'err_pre':>8}{'err_ratio':>11}"
    )
    for ft in sorted(by_ft_global):
        rows = by_ft_global[ft]
        n = len(rows)
        def med(getter):
            xs = sorted(getter(row) for row in rows)
            return xs[len(xs)//2] if xs else 0
        r5_d = med(lambda r: r["during"]["rate_5xx_over_all_spans"])
        r5_p = med(lambda r: r["pre"]["rate_5xx_over_all_spans"])
        re_d = med(lambda r: r["during"]["rate_error_tag_over_all_spans"])
        re_p = med(lambda r: r["pre"]["rate_error_tag_over_all_spans"])
        ratio5 = (r5_d / r5_p) if r5_p > 0 else (float("inf") if r5_d > 0 else 0)
        ratioE = (re_d / re_p) if re_p > 0 else (float("inf") if re_d > 0 else 0)
        ratio5_s = f"{ratio5:.1f}x" if ratio5 != float("inf") else "∞"
        ratioE_s = f"{ratioE:.1f}x" if ratioE != float("inf") else "∞"
        print(
            f"{ft:<35}{n:>3}  "
            f"{100*r5_d:>7.2f}%{100*r5_p:>7.2f}%{ratio5_s:>11}  "
            f"{100*re_d:>7.2f}%{100*re_p:>7.2f}%{ratioE_s:>11}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "trace_features",
            "ood_dir": str(ood_dir),
            "n_samples": len(samples),
            "samples": results,
            "rule_notes": {
                "5xx_definition": "span tag http.status_code starts with '5'",
                "error_tag_definition": "span tag error == True (case-insensitive)",
                "latency_unit": "microseconds (Jaeger native)",
                "window_split": (
                    "spans bucketed by span.startTime vs "
                    "meta.timestamps.inject_start/inject_end (microsecond compare)"
                ),
            },
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


                                                                        
        
                                                                        

def run_stage1(ood_dir: Path, output_path: Path) -> None:
    samples = sorted(
        d for d in ood_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    print(f"Stage 1 — raw-telemetry L-specific signal detection")
    print(f"OOD dir: {ood_dir}")
    print(f"Found {len(samples)} experiments\n")

    print(f"{'#':>2} {'sample':<78} {'gt_layer':<8} {'gt_ft':<32} S1")
    print("-" * 130)
    results = []
    for idx, exp_dir in enumerate(samples, 1):
        r = stage1_for_sample(exp_dir)
        results.append(r)
        print(f"{idx:>2} {r['id']:<78} {r['gt_layer']:<8} {r['gt_ft']:<32} {r['S1']}")

    by_layer: Dict[str, Dict[str, int]] = defaultdict(lambda: {"P": 0, "F": 0})
    by_layer_ft: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"P": 0, "F": 0})
    )
    for r in results:
        by_layer[r["gt_layer"]][r["S1"]] += 1
        by_layer_ft[r["gt_layer"]][r["gt_ft"]][r["S1"]] += 1

    print("\n" + "=" * 80)
    print("Stage 1 aggregate — does raw telemetry contain GT-layer-specific signals?")
    print("=" * 80)
    print(f"{'GT layer':<10}{'Pass':>6}{'Fail':>6}{'Total':>7}{'Pass rate':>12}")
    for layer in sorted(by_layer):
        p, f = by_layer[layer]["P"], by_layer[layer]["F"]
        total = p + f
        rate = f"{100*p/total:.0f}%" if total else "—"
        print(f"{layer:<10}{p:>6}{f:>6}{total:>7}{rate:>12}")

    print("\nPer fault_type breakdown:")
    for layer in sorted(by_layer_ft):
        print(f"  {layer}:")
        for ft in sorted(by_layer_ft[layer]):
            p, f = by_layer_ft[layer][ft]["P"], by_layer_ft[layer][ft]["F"]
            total = p + f
            print(f"    {ft:<35} P={p} F={f} (total={total})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "stage": 1,
            "ood_dir": str(ood_dir),
            "n_samples": len(samples),
            "samples": results,
            "aggregate_by_layer": {k: dict(v) for k, v in by_layer.items()},
            "aggregate_by_layer_ft": {
                layer: {ft: dict(counts) for ft, counts in fts.items()}
                for layer, fts in by_layer_ft.items()
            },
            "rule_sources": {
                "metric_layer_grouping": (
                    "LAYER_METRICS in diagnose_ood_stages.py — derived from "
                    "exp_runs_dataset._METRIC_FILES group comments (each "
                    "channel block has a '# L? <description>' header)."
                ),
                "log_keyword_grouping": (
                    "LAYER_LOG_KEYWORDS in diagnose_ood_stages.py — standard "
                    "K8s/Java/NIC error keywords, broader than signal_extractor."
                ),
                "anomaly_criterion": (
                    "Per L-specific channel: max(during_injection) > "
                    "max(baseline_outside_injection) AND max(during) > 0. "
                    "Uses each channel's own baseline as reference — no "
                    "cross-channel std normalization (which fails for raw "
                    "Prometheus values with mixed units)."
                ),
                "pass_definition": (
                    "Sample passes Stage 1 iff at least one GT-layer-specific "
                    "metric channel is anomalous-during-injection OR at least "
                    "one GT-layer-specific log keyword has any hit."
                ),
            },
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["stage1", "trace_features", "log_timing", "metric_slopes"],
        default="stage1",
        help=(
            "Which experiment to run. 'stage1' = original 5-stage protocol "
            "Stage 1 (raw-telemetry L-specific signal). Others are "
            "data-recovery sub-experiments."
        ),
    )
    parser.add_argument("--ood-dir", type=Path, default=DEFAULT_OOD_DIR)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default chosen per --experiment)",
    )
    args = parser.parse_args()

    default_outputs = {
        "stage1": PROJ_ROOT / "results" / "ood_stage1_audit.json",
        "trace_features": PROJ_ROOT / "results" / "ood_trace_features.json",
        "log_timing": PROJ_ROOT / "results" / "ood_log_timing.json",
        "metric_slopes": PROJ_ROOT / "results" / "ood_metric_slopes.json",
    }
    output_path = args.output or default_outputs[args.experiment]

    if args.experiment == "stage1":
        run_stage1(args.ood_dir, output_path)
    elif args.experiment == "trace_features":
        run_trace_features(args.ood_dir, output_path)
    elif args.experiment == "log_timing":
        run_log_timing(args.ood_dir, output_path)
    elif args.experiment == "metric_slopes":
        run_metric_slopes(args.ood_dir, output_path)
    else:
        print(f"Experiment {args.experiment} not yet implemented")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
