"""
Trace-centric data alignment utilities.

Implements:
1. Hard alignment: logs matched by exact trace_id
2. Soft alignment: metrics matched by time window with relative positions
3. Trace graph construction from span data (supports SkyWalking and Jaeger formats)
4. Relative time position computation for Gaussian attention
"""
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


@dataclass
class MetricTimeSeries:
    """A single metric time series with optional relative positions."""

    name: str
    timestamps: np.ndarray
    values: np.ndarray
    relative_positions: Optional[np.ndarray] = None


def compute_relative_time_positions(
    metric_timestamps: Union[np.ndarray, Tensor],
    anchor_timestamp: float,
) -> Union[np.ndarray, Tensor]:
    """
    Compute |t_metric - t_trace| for Gaussian attention.

    Args:
        metric_timestamps: Array of metric timestamps (seconds).
        anchor_timestamp: The trace anchor timestamp (seconds).

    Returns:
        Absolute time distances (same type as input).
    """
    if isinstance(metric_timestamps, Tensor):
        return torch.abs(metric_timestamps - anchor_timestamp)
    return np.abs(metric_timestamps - anchor_timestamp)


def align_logs_by_trace_id(logs: List[Dict], trace_id: str) -> List[Dict]:
    """
    Hard alignment: select logs matching the given trace_id.

    Supports multiple field names (traceid, traceId, trace_id) and
    case-insensitive matching.

    Args:
        logs: List of log dicts, each may contain a trace ID field.
        trace_id: The trace ID to match.

    Returns:
        Filtered and time-sorted list of matching logs.
    """
    trace_id_lower = trace_id.lower()
    aligned = []

    for log in logs:
        log_trace_id = None
        for key in ("traceid", "traceId", "trace_id"):
            if key in log:
                log_trace_id = log[key]
                break

        if log_trace_id is None:
            continue

        if str(log_trace_id).lower() == trace_id_lower:
            aligned.append(log)

    aligned.sort(key=lambda x: x.get("timestamp", x.get("timestamp_ns", 0)))
    return aligned


def align_metrics_by_time_window(
    metrics: List[MetricTimeSeries],
    center_timestamp: float,
    window_seconds: float,
) -> List[MetricTimeSeries]:
    """
    Soft alignment: select metric points within a time window.

    Args:
        metrics: List of metric time series.
        center_timestamp: Center of the time window (seconds).
        window_seconds: Half-width of the window (seconds).

    Returns:
        List of MetricTimeSeries filtered to the window, with
        relative_positions computed.
    """
    aligned = []

    for m in metrics:
        mask = np.abs(m.timestamps - center_timestamp) <= window_seconds
        ts_filtered = m.timestamps[mask]
        vals_filtered = m.values[mask]
        rel_pos = (
            np.abs(ts_filtered - center_timestamp) if len(ts_filtered) > 0 else np.array([])
        )

        aligned.append(
            MetricTimeSeries(
                name=m.name,
                timestamps=ts_filtered,
                values=vals_filtered,
                relative_positions=rel_pos,
            )
        )

    return aligned


def preprocess_jaeger_spans(jaeger_spans: List[Dict]) -> List[Dict]:
    'English text removed.'
    if not jaeger_spans:
        return []

                                            
    has_children = set()
    for s in jaeger_spans:
        for ref in s.get("references", []):
            if ref.get("refType") == "CHILD_OF":
                has_children.add(ref["spanID"])

                                   
    depth_map: Dict[str, int] = {}
    for s in jaeger_spans:
        parent_refs = [r for r in s.get("references", [])
                       if r.get("refType") == "CHILD_OF"]
        if not parent_refs:
            depth_map[s["spanID"]] = 0

    changed = True
    while changed:
        changed = False
        for s in jaeger_spans:
            sid = s["spanID"]
            if sid in depth_map:
                continue
            for ref in s.get("references", []):
                if ref.get("refType") == "CHILD_OF" and ref["spanID"] in depth_map:
                    depth_map[sid] = depth_map[ref["spanID"]] + 1
                    changed = True
                    break

    preprocessed = []
    for s in jaeger_spans:
        sid = s["spanID"]

                              
        tags_dict = {t["key"]: t["value"] for t in s.get("tags", [])}
        is_error = bool(tags_dict.get("error", False))
        if not is_error:
            status = tags_dict.get("http.status_code", 0)
            try:
                if int(status) >= 400:
                    is_error = True
            except (TypeError, ValueError):
                pass

            
        depth = depth_map.get(sid, 0)

                                              
        parent_refs = [r for r in s.get("references", [])
                       if r.get("refType") == "CHILD_OF"]
        if not parent_refs:
            span_type = "Entry"
        elif sid not in has_children:
            span_type = "Exit"
        else:
            span_type = "Local"

        preprocessed.append({
            "spanID":      sid,
            "references":  s.get("references", []),
            "duration":    s.get("duration", 0),                  
            "is_error":    is_error,
            "depth":       float(depth),
            "type":        span_type,
        })

    return preprocessed


def build_trace_graph(
    spans: List[Dict],
    duration_stats: Optional[Dict[str, float]] = None,
) -> Tuple[Tensor, Tensor, List[str]]:
    """
    Build adjacency matrix and feature matrix from a list of spans.

    Supports both SkyWalking format (node_id, parent_node_id) and
    Jaeger format (spanID, parentSpanID / references).

    Span features extracted (dim=6):
        [duration_norm, is_error, depth_norm, is_entry, is_exit, is_local]

    Duration normalization:
    - If duration_stats is provided: log1p + global z-score normalization.
      Preserves absolute duration scale across traces.
    - Otherwise: per-trace max normalization (legacy fallback).

    Args:
        spans: List of span dicts.
        duration_stats: Dict with 'mean'/'std' of log1p(raw_duration_us)
            for z-score normalization. Computed from training set.

    Returns:
        (adjacency, features, span_ids):
        - adjacency: [n, n] float tensor with self-loops and parent-child edges
        - features: [n, 6] float tensor of span features
        - span_ids: list of span ID strings
    """
    if not spans:
        return torch.zeros(0, 0), torch.zeros(0, 0), []

    # --- Collect span IDs ---
    span_ids: List[str] = []
    for s in spans:
        if "node_id" in s and s["node_id"] is not None:
            sid = s["node_id"]
        elif "spanID" in s:
            sid = s["spanID"]
        else:
            sid = str(s.get("span_id", len(span_ids)))
        span_ids.append(str(sid))

    n = len(spans)
    sid_to_idx = {sid: i for i, sid in enumerate(span_ids)}

    # --- Build adjacency (with self-loops) ---
    adj = torch.eye(n)

    for s in spans:
        # Determine child and parent IDs
        if "node_id" in s and s["node_id"] is not None:
            child_id = s["node_id"]
        elif "spanID" in s:
            child_id = s["spanID"]
        else:
            child_id = str(s.get("span_id", ""))

        child_id = str(child_id)

        parent_id = None
        if "parent_node_id" in s and s["parent_node_id"] is not None:
            parent_id = str(s["parent_node_id"])
        elif "parentSpanID" in s and s["parentSpanID"] is not None:
            parent_id = str(s["parentSpanID"])
        else:
            # Jaeger reference format
            for ref in s.get("references", []):
                if ref.get("refType") == "CHILD_OF":
                    parent_id = str(ref.get("spanID"))
                    break

        if parent_id and parent_id in sid_to_idx:
            child_idx = sid_to_idx[child_id]
            parent_idx = sid_to_idx[parent_id]
            adj[parent_idx, child_idx] = 1.0
            adj[child_idx, parent_idx] = 1.0  # Bidirectional

    # --- Extract features ---
    features = []
    for s in spans:
        # Duration
        if "duration_ms" in s:
            duration = float(s["duration_ms"])
        elif "duration" in s:
            duration = float(s["duration"])
        else:
            duration = 0.0

        # Error flag
        if "is_error" in s:
            is_error = float(bool(s["is_error"]))
        elif "isError" in s:
            is_error = float(bool(s["isError"]))
        else:
            is_error = 0.0

        # Depth
        depth = float(s.get("depth", 0))

        # Span type (SkyWalking)
        span_type = s.get("type", "")
        is_entry = float(span_type == "Entry")
        is_exit = float(span_type == "Exit")
        is_local = float(span_type == "Local")

        features.append([duration, is_error, depth, is_entry, is_exit, is_local])

    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Normalize duration and depth
    if n > 0:
        if duration_stats is not None:
            # log1p + global z-score normalization
            # Preserves absolute duration scale (e.g. 490k vs 23k us)
            features_tensor[:, 0] = torch.log1p(features_tensor[:, 0])
            mean = duration_stats.get("mean", 0.0)
            std = duration_stats.get("std", 1.0)
            features_tensor[:, 0] = (features_tensor[:, 0] - mean) / (std + 1e-8)
        else:
            # Fallback: per-trace max normalization
            max_dur = features_tensor[:, 0].max()
            if max_dur > 0:
                features_tensor[:, 0] = features_tensor[:, 0] / max_dur
        max_depth = features_tensor[:, 2].max()
        if max_depth > 0:
            features_tensor[:, 2] = features_tensor[:, 2] / max_depth

    return adj, features_tensor, span_ids


# ---------------------------------------------------------------------------
# Rich span feature extraction (T1 + T2 of the trace remediation roadmap).
#
# These helpers are *additive* — they do not modify the existing
# preprocess_jaeger_spans / build_trace_graph behaviour. They are only
# invoked when the dataset is constructed with enable_rich_span_features=True,
# so the default code path remains byte-identical to pre-modification.
# ---------------------------------------------------------------------------


def stable_hash_bucket(text: str, num_buckets: int) -> int:
    """Deterministic hash of `text` to [0, num_buckets).

    Uses MD5 instead of Python's built-in hash() because PYTHONHASHSEED
    randomizes hash() per process by default — that breaks reproducibility
    across DataLoader workers and across runs.
    """
    if num_buckets <= 0:
        return 0
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % num_buckets


def status_code_to_class(status_code) -> int:
    """Bucket an HTTP status code to {0=none, 1=2xx, 2=4xx, 3=5xx}."""
    try:
        c = int(status_code)
    except (TypeError, ValueError):
        return 0
    if 200 <= c < 300:
        return 1
    if 400 <= c < 500:
        return 2
    if c >= 500:
        return 3
    return 0


def compute_critical_path_set(jaeger_spans: List[Dict]) -> set:
    """Return the set of span IDs that lie on the critical path of the trace.

    Algorithm: starting from each root span (no CHILD_OF parent), walk down the
    call tree picking the child with the latest ``startTime + duration``
    (the bottleneck child) at every level. Every span visited on the way down
    is on the critical path. With multiple roots we take the union of their
    per-root critical paths.

    This matches the standard "longest dependency-aware path" definition used
    by trace-analysis tooling: spans not on the path either ran in parallel
    or finished strictly faster than a sibling.
    """
    if not jaeger_spans:
        return set()

    by_id: Dict[str, Dict] = {}
    children_map: Dict[str, List[str]] = {}
    parent_count: Dict[str, int] = {}

    for s in jaeger_spans:
        sid = s.get("spanID")
        if sid is None:
            continue
        by_id[sid] = s
        parent_count.setdefault(sid, 0)

    for s in jaeger_spans:
        sid = s.get("spanID")
        for ref in s.get("references", []):
            if ref.get("refType") == "CHILD_OF":
                parent_id = ref.get("spanID")
                if parent_id is None:
                    continue
                children_map.setdefault(parent_id, []).append(sid)
                parent_count[sid] = parent_count.get(sid, 0) + 1

    def _end_time(span_id):
        s = by_id.get(span_id)
        if s is None:
            return -1
        return int(s.get("startTime", 0) or 0) + int(s.get("duration", 0) or 0)

    roots = [sid for sid in by_id if parent_count.get(sid, 0) == 0]

    on_path: set = set()
    for root_id in roots:
        cur = root_id
        # Loop with cycle protection — Jaeger trees should not have cycles
        # but a malformed trace must not hang the worker.
        seen: set = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            on_path.add(cur)
            children = children_map.get(cur, [])
            if not children:
                break
            cur = max(children, key=_end_time)
    return on_path


# OTel status code mapping (per OTel spec): UNSET / OK / ERROR.
_OTEL_STATUS_TO_CLASS = {"UNSET": 0, "OK": 1, "ERROR": 2}


def extract_rich_span_features(
    jaeger_spans: List[Dict],
    processes_map: Optional[Dict[str, Dict]] = None,
    num_service_buckets: int = 64,
    num_operation_buckets: int = 1024,
    num_error_type_buckets: int = 32,
) -> Dict[str, Tensor]:
    """Extract categorical span features for the rich-feature trace encoder.

    Reads operation_name (Jaeger ``operationName``), service_name (resolved
    via ``processID`` against the trace-level ``processes`` map), HTTP
    status from span tags, plus the round2-only error tags
    (``error``, ``error.type``, ``otel.status_code``).

    Args:
        jaeger_spans: raw Jaeger spans (with ``processID``, ``operationName``,
            ``tags``, ``references``).
        processes_map: trace-level ``processes`` dict from the Jaeger JSON,
            shape ``{"p1": {"serviceName": "..."}}``.
        num_service_buckets: hash bucket count for service_name.
        num_operation_buckets: hash bucket count for operation_name.
        num_error_type_buckets: hash bucket count for ``error.type`` strings.

    Returns:
        Dict with long tensors of shape ``[num_spans]``:
        - ``service_ids``: hashed service_name bucket
        - ``operation_ids``: hashed operation_name bucket
        - ``status_classes``: 0=none / 1=2xx / 2=4xx / 3=5xx (HTTP status)
        - ``is_root``: 1 if the span has no CHILD_OF parent
        - ``is_leaf``: 1 if the span has no CHILD_OF children
        - ``is_on_critical_path``: 1 if span lies on the dependency-aware
          critical path of the trace
        - ``error_flag``: 1 if the OTel ``error`` tag is true (round2 only)
        - ``error_type_ids``: hashed ``error.type`` bucket; empty string
          maps to a deterministic "no error" bucket
        - ``otel_status_classes``: 0=UNSET / 1=OK / 2=ERROR
    """
    if not jaeger_spans:
        return {
            "service_ids": torch.zeros(0, dtype=torch.long),
            "operation_ids": torch.zeros(0, dtype=torch.long),
            "status_classes": torch.zeros(0, dtype=torch.long),
            "is_root": torch.zeros(0, dtype=torch.long),
            "is_leaf": torch.zeros(0, dtype=torch.long),
            "is_on_critical_path": torch.zeros(0, dtype=torch.long),
            "error_flag": torch.zeros(0, dtype=torch.long),
            "error_type_ids": torch.zeros(0, dtype=torch.long),
            "otel_status_classes": torch.zeros(0, dtype=torch.long),
        }

    processes_map = processes_map or {}

    # Span IDs that appear as a CHILD_OF parent of some other span — i.e. they have children.
    has_children: set = set()
    for s in jaeger_spans:
        for ref in s.get("references", []):
            if ref.get("refType") == "CHILD_OF":
                has_children.add(ref.get("spanID"))

    # T3: critical path set (computed once per trace, then looked up per span).
    critical_path_set = compute_critical_path_set(jaeger_spans)

    service_ids: List[int] = []
    operation_ids: List[int] = []
    status_classes: List[int] = []
    is_root_flags: List[int] = []
    is_leaf_flags: List[int] = []
    is_on_critical_path_flags: List[int] = []
    error_flags: List[int] = []
    error_type_ids: List[int] = []
    otel_status_classes: List[int] = []

    for s in jaeger_spans:
        # Service name via processID lookup; "unknown" when missing so the
        # hash is still deterministic instead of NaN.
        process_id = s.get("processID", "")
        proc_info = processes_map.get(process_id, {}) if isinstance(processes_map, dict) else {}
        service_name = proc_info.get("serviceName", "unknown") or "unknown"

        operation_name = s.get("operationName", "unknown") or "unknown"

        # Single tag scan to pick up HTTP status, error flag, error.type, otel.status_code.
        status_code = 0
        error_flag = 0
        error_type_str = ""
        otel_status_str = "UNSET"
        for tag in s.get("tags", []):
            key = tag.get("key")
            if key == "http.response.status_code":
                status_code = tag.get("value", 0)
            elif key == "error":
                v = tag.get("value")
                if v is True or (isinstance(v, str) and v.lower() == "true"):
                    error_flag = 1
            elif key == "error.type":
                error_type_str = str(tag.get("value", "") or "")
            elif key == "otel.status_code":
                otel_status_str = str(tag.get("value", "UNSET") or "UNSET").upper()

        parent_refs = [
            r for r in s.get("references", [])
            if r.get("refType") == "CHILD_OF"
        ]
        is_root = 1 if not parent_refs else 0
        is_leaf = 1 if s.get("spanID") not in has_children else 0

        service_ids.append(stable_hash_bucket(str(service_name), num_service_buckets))
        operation_ids.append(stable_hash_bucket(str(operation_name), num_operation_buckets))
        status_classes.append(status_code_to_class(status_code))
        is_root_flags.append(is_root)
        is_leaf_flags.append(is_leaf)
        is_on_critical_path_flags.append(
            1 if s.get("spanID") in critical_path_set else 0
        )
        error_flags.append(error_flag)
        error_type_ids.append(stable_hash_bucket(error_type_str, num_error_type_buckets))
        otel_status_classes.append(_OTEL_STATUS_TO_CLASS.get(otel_status_str, 0))

    return {
        "service_ids": torch.tensor(service_ids, dtype=torch.long),
        "operation_ids": torch.tensor(operation_ids, dtype=torch.long),
        "status_classes": torch.tensor(status_classes, dtype=torch.long),
        "is_root": torch.tensor(is_root_flags, dtype=torch.long),
        "is_leaf": torch.tensor(is_leaf_flags, dtype=torch.long),
        "is_on_critical_path": torch.tensor(is_on_critical_path_flags, dtype=torch.long),
        "error_flag": torch.tensor(error_flags, dtype=torch.long),
        "error_type_ids": torch.tensor(error_type_ids, dtype=torch.long),
        "otel_status_classes": torch.tensor(otel_status_classes, dtype=torch.long),
    }
