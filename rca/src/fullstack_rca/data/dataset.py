"""
MicroServiceDataset: Trace-centric multimodal dataset for AnoMod.

Loads SkyWalking traces from AnoMod's TT_data directory structure,
extracts span features and adjacency matrices, tokenizes pseudo-log
messages from span metadata, and provides placeholder metrics.

Expected directory layout (AnoMod):
    data_dir/
        trace_data/
            {scenario_name}/
                {scenario}_traces_*.json
        log_data/
            {scenario_name}_*/
                {pod_name}/*.log
        metric_data/
            {scenario_name}/
                *.csv

Scenario naming convention:
    Normal_case_em_*       -> Normal (no fault)
    Lv_P_*                 -> L1 Infrastructure (CPU, DISKIO, NETLOSS)
    Lv_S_*                 -> L2 Network (DNSFAIL, HTTPABORT, KILLPOD)
    Lv_D_*                 -> L3 Platform (CONNECTION_POOL, TRANSACTION, cachelimit)
    Lv_C_*                 -> L4 Application (security_check, exception_injection, travel_detail_failure)
"""
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .alignment import (
    MetricTimeSeries,
    align_logs_by_trace_id,
    align_metrics_by_time_window,
    build_trace_graph,
    compute_relative_time_positions,
)

logger = logging.getLogger(__name__)

# Scenario name prefix -> (layer string, layer index)
LAYER_PREFIX_MAP = {
    "Lv_P": ("L1", 0),   # Physical / Infrastructure
    "Lv_S": ("L2", 1),   # Service / Network
    "Lv_D": ("L3", 2),   # Data / Platform
    "Lv_C": ("L4", 3),   # Code / Application
}


FAULT_TYPE_TO_ID = {
    "cpu_pressure": 0,
    "disk_io_saturation": 1,
    "memory_pressure": 2,
    "connection_reset": 3,   # round3: replaced dns_failure at this slot
    "network_latency": 4,
    "packet_loss": 5,
    "pod_kill": 6,
    "liveness_probe_failure": 7,
    "resource_quota_exhaustion": 8,
    "jvm_memory_pressure": 9,
    "database_connection_exhaustion": 10,
    "http_error_injection": 11,
}

# SocialNetwork (round_sn1) 12-class taxonomy. Ordered so id // 3 == layer index
# (L1=0-2, L2=3-5, L3=6-8, L4=9-11), matching the TT convention the layer-distance
# loss weighting relies on.
SN_FAULT_TYPE_TO_ID = {
    "cpu_pressure": 0,
    "disk_io_saturation": 1,
    "memory_pressure": 2,
    "dns_failure": 3,
    "network_latency": 4,
    "packet_loss": 5,
    "container_kill": 6,
    "health_endpoint_block": 7,
    "cgroup_quota_squeeze": 8,
    "container_memory_pressure": 9,
    "mongo_connection_exhaustion": 10,
    "http_error_injection": 11,
}

_FAULT_TYPE_MAPS = {"tt": FAULT_TYPE_TO_ID, "sn": SN_FAULT_TYPE_TO_ID}

# Also map AnoMod-style names (used by MicroServiceDataset / TT_data)
# Verified against rca/config/label_mapping.yaml
_ANOMOD_FT_ALIASES = {
    "CPU_preserve": "cpu_pressure",
    "DISKIO_preserve": "disk_io_saturation",
    "MEM_preserve": "memory_pressure",
    "NETLOSS_preserve": "packet_loss",
    # NOTE: round3 dropped dns_failure from the canonical taxonomy (replaced by
    # connection_reset), so this cross-domain alias now resolves to an unknown
    # class (get_fault_type_id → -1). Kept for the deferred cross-domain path.
    "DNSFAIL_preserve_no_order": "dns_failure",
    "NETWORK_DELAY_preserve_no_order": "network_latency",
    "PACKET_LOSS_preserve_no_order": "packet_loss",
    "HTTPABORT_preserve": "http_error_injection",
    "KILLPOD_preserve": "pod_kill",
    "oomkilled": "pod_kill",
    "cachelimit": "resource_quota_exhaustion",
    "CONNECTION_POOL_exhaustion": "database_connection_exhaustion",
    "TRANSACTION_timeout": "database_connection_exhaustion",
    "exception_injection": "http_error_injection",
    "travel_detail_failure": "http_error_injection",
    "security_check": "http_error_injection",
}


def get_fault_type_id(root_cause: str, system: str = "tt") -> int:
    """Map a root_cause string to a fault_type_id (0-11), or -1 if unknown.

    system: "tt" (default, canonical TT taxonomy + AnoMod aliases) or "sn"
    (SocialNetwork round_sn1 taxonomy). Default keeps every existing caller
    byte-identical.
    """
    table = _FAULT_TYPE_MAPS.get(system, FAULT_TYPE_TO_ID)
    if system == "tt":
        canonical = _ANOMOD_FT_ALIASES.get(root_cause, root_cause)
    else:
        canonical = root_cause
    return table.get(canonical, -1)


def _parse_scenario_label(scenario_name: str) -> Tuple[Optional[str], int, str]:
    """Parse scenario directory name to extract fault layer and root cause type.

    Returns:
        (layer, layer_idx, root_cause_type)
        layer is None and layer_idx is -1 for normal scenarios.
    """
    if scenario_name.startswith("Normal"):
        return None, -1, "normal"

    for prefix, (layer, layer_idx) in LAYER_PREFIX_MAP.items():
        if scenario_name.startswith(prefix):
            # Remove prefix + underscore, then strip timestamp suffix
            rest = scenario_name[len(prefix) + 1:]
            parts = rest.split("_")
            rca_parts = []
            for p in parts:
                # Timestamp part starts with 8+ digits (20251103T...)
                if len(p) >= 8 and p[:8].isdigit():
                    break
                rca_parts.append(p)
            root_cause = "_".join(rca_parts) if rca_parts else rest
            return layer, layer_idx, root_cause

    return None, -1, "unknown"


@dataclass
class MultimodalSample:
    """A single trace-centric multimodal sample."""

    trace_id: str
    span_features: Tensor           # [num_spans, span_dim]
    trace_adjacency: Tensor         # [num_spans, num_spans]
    log_tokens: Tensor              # [num_logs, seq_len]
    log_attention_mask: Tensor      # [num_logs, seq_len]
    metric_values: Tensor           # [num_metrics, metric_seq_len]
    relative_time_positions: Tensor # [metric_seq_len]
    label: int = 0
    fault_layer: int = -1           # 0-3 for L1-L4, -1 for normal
    layer: Optional[str] = None     # "L1" .. "L4"
    root_cause_type: Optional[str] = None
    num_logs: int = 0
    fault_type_id: int = -1         # 0-11 for 12 fault types, -1 for normal/unknown
    # Rich span features (populated only when the dataset is constructed with
    # enable_rich_span_features=True). When all are None the collate_fn does
    # not emit them, so the batch dict and downstream behaviour are
    # byte-identical to pre-modification.
    span_service_ids: Optional[Tensor] = None      # [num_spans] long
    span_operation_ids: Optional[Tensor] = None    # [num_spans] long
    span_status_classes: Optional[Tensor] = None   # [num_spans] long, 0..3
    span_is_root: Optional[Tensor] = None          # [num_spans] long, 0/1
    span_is_leaf: Optional[Tensor] = None          # [num_spans] long, 0/1
    span_is_on_critical_path: Optional[Tensor] = None  # [num_spans] long, 0/1 (T3)
    # Round2-only OTel error tags (populated alongside the rich span features
    # when the dataset is constructed with enable_rich_span_features=True).
    span_error_flag: Optional[Tensor] = None              # [num_spans] long, 0/1
    span_error_type_ids: Optional[Tensor] = None          # [num_spans] long, hash bucket
    span_otel_status_classes: Optional[Tensor] = None     # [num_spans] long, 0=UNSET/1=OK/2=ERROR


class MicroServiceDataset(Dataset):
    """Trace-centric multimodal dataset for AnoMod.

    Each sample corresponds to one distributed trace, with associated
    span graph features, pseudo-log tokens, and placeholder metrics.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        experiment_dir: Optional[str] = None,
        metric_window_seconds: float = 60.0,
        log_window_seconds: float = 60.0,
        max_logs_per_trace: int = 20,
        num_metrics: int = 10,
        max_metric_points: int = 60,
        max_traces_per_scenario: int = 100,
        max_spans_per_trace: int = 50,
        log_max_length: int = 64,
        include_labels: bool = True,
        tokenizer_name: str = "distilbert-base-uncased",
        duration_stats: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir or experiment_dir)
        self.metric_window_seconds = metric_window_seconds
        self.log_window_seconds = log_window_seconds
        self.max_logs_per_trace = max_logs_per_trace
        self.num_metrics = num_metrics
        self.max_metric_points = max_metric_points
        self.max_traces_per_scenario = max_traces_per_scenario
        self.max_spans_per_trace = max_spans_per_trace
        self.log_max_length = log_max_length
        self.include_labels = include_labels

        # Duration stats for log1p + z-score normalization (set by training script)
        self.duration_stats = duration_stats

        # Lazy-init tokenizer
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name

        # Index: list of dicts describing each sample
        self._samples_index: List[Dict] = []
        # Cache: scenario_name -> list of trace dicts
        self._trace_cache: Dict[str, list] = {}
        # Cache: scenario_name -> parsed raw log entries
        self._log_cache: Dict[str, List[Dict[str, Any]]] = {}
        # Cache: scenario_name -> metric time series
        self._metric_cache: Dict[str, List[MetricTimeSeries]] = {}

        self._trace_id_regex = re.compile(r"(?:traceid|trace_id|traceId)\s*[:=]\s*([A-Za-z0-9._-]+)")
        self._log_ts_regex = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
        self._log_tz = timezone(timedelta(hours=8))

        self._load_scenarios()

    # ------------------------------------------------------------------
    # Tokenizer (lazy-loaded to avoid import cost when not needed)
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            except Exception:
                self._tokenizer = "fallback"
        return self._tokenizer

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_scenarios(self):
        """Scan trace_data/ for scenario directories and build sample index."""
        trace_data_dir = self.data_dir / "trace_data"

        if not trace_data_dir.exists():
            raise FileNotFoundError(
                f"trace_data directory not found under {self.data_dir}. "
                f"Expected AnoMod layout with trace_data/ subdirectory."
            )

        scenario_dirs = sorted(d for d in trace_data_dir.iterdir() if d.is_dir())
        logger.info("Found %d scenarios under %s", len(scenario_dirs), trace_data_dir)

        for scenario_dir in scenario_dirs:
            scenario_name = scenario_dir.name
            layer, layer_idx, root_cause = _parse_scenario_label(scenario_name)

            # Find trace JSON files in this scenario
            json_files = sorted(scenario_dir.glob("*.json"))
            if not json_files:
                logger.warning("No JSON files in %s, skipping", scenario_dir)
                continue

            # Load the first (usually only) JSON file
            trace_file = json_files[0]
            try:
                with open(trace_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", trace_file, e)
                continue

            traces = data.get("traces", [])
            traces = traces[: self.max_traces_per_scenario]

            self._trace_cache[scenario_name] = traces
            logger.info(
                "  %s: %d traces loaded (layer=%s, rca=%s)",
                scenario_name,
                len(traces),
                layer,
                root_cause,
            )

            for i, trace in enumerate(traces):
                summary = trace.get("summary", {})
                trace_id = summary.get("trace_id", f"{scenario_name}_{i}")

                self._samples_index.append(
                    {
                        "scenario_name": scenario_name,
                        "trace_idx": i,
                        "trace_id": trace_id,
                        "layer": layer,
                        "layer_idx": layer_idx,
                        "root_cause": root_cause,
                        "start_timestamp_ms": summary.get("start_timestamp_ms", 0),
                    }
                )

        logger.info("Total samples indexed: %d", len(self._samples_index))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples_index)

    def __getitem__(self, idx: int) -> MultimodalSample:
        info = self._samples_index[idx]
        scenario_name = info["scenario_name"]
        trace_idx = info["trace_idx"]

        trace = self._trace_cache[scenario_name][trace_idx]
        spans = trace.get("spans", [])

        # Limit spans
        if len(spans) > self.max_spans_per_trace:
            spans = spans[: self.max_spans_per_trace]

        # Build trace graph -> (adjacency, features, span_ids)
        adjacency, span_features, span_ids = build_trace_graph(
            spans,
            duration_stats=self.duration_stats,
        )

        # Handle empty traces
        if len(span_ids) == 0:
            adjacency = torch.zeros(1, 1)
            span_features = torch.zeros(1, 6)

        anchor_timestamp_ms = int(info.get("start_timestamp_ms", 0) or 0)
        if anchor_timestamp_ms <= 0 and spans:
            anchor_timestamp_ms = int(spans[0].get("start_timestamp_ms", 0) or 0)
        anchor_timestamp_s = anchor_timestamp_ms / 1000.0 if anchor_timestamp_ms > 0 else 0.0

        # Build aligned log texts from real logs and span events
        log_texts = self._collect_aligned_logs(
            scenario_name=scenario_name,
            trace_id=info["trace_id"],
            spans=spans,
            anchor_timestamp_s=anchor_timestamp_s,
        )
        log_tokens, log_mask = self._tokenize_logs(log_texts)

        # Build aligned metrics from real metric CSV
        metric_values, relative_positions = self._build_aligned_metrics(
            scenario_name=scenario_name,
            anchor_timestamp_s=anchor_timestamp_s,
        )

        return MultimodalSample(
            trace_id=info["trace_id"],
            span_features=span_features,
            trace_adjacency=adjacency,
            log_tokens=log_tokens,
            log_attention_mask=log_mask,
            metric_values=metric_values,
            relative_time_positions=relative_positions,
            label=max(0, info["layer_idx"]),
            fault_layer=info["layer_idx"],
            layer=info["layer"],
            root_cause_type=info["root_cause"],
            num_logs=len(log_texts),
            fault_type_id=get_fault_type_id(info["root_cause"]),
        )

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_log_texts(self, spans: List[Dict]) -> List[str]:
        """Build pseudo-log messages from span metadata."""
        texts = []
        for span in spans:
            parts = [
                span.get("service_code", "unknown"),
                span.get("endpoint_name", "unknown"),
                span.get("component", ""),
                span.get("layer", ""),
            ]
            if span.get("is_error"):
                parts.append("ERROR")

            tags_map = span.get("tags_map", {})
            if "http.method" in tags_map:
                parts.append(tags_map["http.method"])
            if "db.type" in tags_map:
                parts.append(tags_map["db.type"])
            if "db.statement" in tags_map:
                parts.append(tags_map["db.statement"][:80])

            texts.append(" ".join(p for p in parts if p))

        return texts[: self.max_logs_per_trace]

    def _extract_trace_id(self, text: str) -> Optional[str]:
        """Extract trace ID from a log line if present."""
        m = self._trace_id_regex.search(text)
        return m.group(1) if m else None

    def _parse_log_timestamp_ms(self, line: str) -> Optional[int]:
        """Parse log line timestamp (assumed UTC+8 in AnoMod logs)."""
        m = self._log_ts_regex.match(line)
        if not m:
            return None
        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=self._log_tz)
        return int(dt.timestamp() * 1000)

    def _normalize_service_name(self, pod_name: str) -> str:
        """
        Normalize pod name to service name.
        Example: ts-order-service-86d6f7876-6ghzm -> ts-order-service
        """
        parts = pod_name.split("-")
        if len(parts) >= 3 and re.fullmatch(r"[0-9a-f]{8,}", parts[-2]):
            return "-".join(parts[:-2])
        return pod_name

    def _load_scenario_logs(self, scenario_name: str) -> List[Dict[str, Any]]:
        """Load and parse raw logs for a scenario."""
        if scenario_name in self._log_cache:
            return self._log_cache[scenario_name]

        scenario_log_dir = self.data_dir / "log_data" / scenario_name
        entries: List[Dict[str, Any]] = []

        if not scenario_log_dir.exists():
            self._log_cache[scenario_name] = entries
            return entries

        def flush_entry(
            ts_ms: Optional[int],
            service: str,
            lines: List[str],
        ) -> None:
            if ts_ms is None or not lines:
                return
            message = "\n".join(lines).strip()
            if not message:
                return
            entries.append(
                {
                    "timestamp": ts_ms / 1000.0,
                    "timestamp_ms": ts_ms,
                    "service": service,
                    "message": message,
                    "traceid": self._extract_trace_id(message),
                }
            )

        for pod_dir in sorted(p for p in scenario_log_dir.iterdir() if p.is_dir()):
            service_name = self._normalize_service_name(pod_dir.name)
            for log_file in sorted(pod_dir.glob("*.log")):
                current_ts_ms: Optional[int] = None
                current_lines: List[str] = []
                with open(log_file, "r", errors="ignore") as f:
                    for raw_line in f:
                        line = raw_line.rstrip("\n")
                        ts_ms = self._parse_log_timestamp_ms(line)
                        if ts_ms is not None:
                            flush_entry(current_ts_ms, service_name, current_lines)
                            current_ts_ms = ts_ms
                            current_lines = [line]
                        elif current_ts_ms is not None:
                            # Keep stack trace / continuation lines as same event
                            current_lines.append(line)
                flush_entry(current_ts_ms, service_name, current_lines)

        entries.sort(key=lambda x: x["timestamp"])
        self._log_cache[scenario_name] = entries
        return entries

    def _load_scenario_metrics(self, scenario_name: str) -> List[MetricTimeSeries]:
        """Load metric CSV and build per-metric time series cache."""
        if scenario_name in self._metric_cache:
            return self._metric_cache[scenario_name]

        scenario_metric_dir = self.data_dir / "metric_data" / scenario_name
        csv_files = sorted(scenario_metric_dir.glob("*.csv"))
        series_list: List[MetricTimeSeries] = []

        if not csv_files:
            self._metric_cache[scenario_name] = series_list
            return series_list

        metric_file = csv_files[0]
        try:
            df = pd.read_csv(
                metric_file,
                usecols=["metric_name", "timestamp", "value"],
            )
        except Exception as e:
            logger.warning("Failed to load metrics from %s: %s", metric_file, e)
            self._metric_cache[scenario_name] = series_list
            return series_list

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["metric_name", "timestamp", "value"])

        if df.empty:
            self._metric_cache[scenario_name] = series_list
            return series_list

        top_metric_names = (
            df["metric_name"]
            .value_counts()
            .head(self.num_metrics)
            .index
            .tolist()
        )

        for metric_name in top_metric_names:
            metric_df = df[df["metric_name"] == metric_name][["timestamp", "value"]]
            metric_df = (
                metric_df.groupby("timestamp", as_index=False)["value"]
                .mean()
                .sort_values("timestamp")
            )
            series_list.append(
                MetricTimeSeries(
                    name=metric_name,
                    timestamps=metric_df["timestamp"].to_numpy(dtype=np.float64),
                    values=metric_df["value"].to_numpy(dtype=np.float32),
                )
            )

        # Keep metric tensor shape stable even when fewer metrics are available.
        while len(series_list) < self.num_metrics:
            series_list.append(
                MetricTimeSeries(
                    name=f"missing_metric_{len(series_list)}",
                    timestamps=np.array([], dtype=np.float64),
                    values=np.array([], dtype=np.float32),
                )
            )

        self._metric_cache[scenario_name] = series_list
        return series_list

    def _extract_span_event_logs(self, spans: List[Dict]) -> List[Dict[str, Any]]:
        """Extract structured log events carried by trace spans."""
        events: List[Dict[str, Any]] = []
        for span in spans:
            trace_id = str(span.get("trace_id", ""))
            service = str(span.get("service_code", ""))
            endpoint = str(span.get("endpoint_name", ""))
            for item in span.get("logs", []):
                if not isinstance(item, dict):
                    continue
                time_ms = int(item.get("time_ms", 0) or 0)
                data = item.get("data", {})
                if isinstance(data, dict):
                    details = " ".join(f"{k}={v}" for k, v in data.items())
                else:
                    details = str(data)
                message = f"{service} {endpoint} {details}".strip()
                events.append(
                    {
                        "traceid": trace_id,
                        "timestamp": (time_ms / 1000.0) if time_ms > 0 else 0.0,
                        "message": message,
                        "service": service,
                    }
                )
        return events

    def _collect_aligned_logs(
        self,
        scenario_name: str,
        trace_id: str,
        spans: List[Dict],
        anchor_timestamp_s: float,
    ) -> List[str]:
        """Collect real logs aligned to the trace anchor."""
        span_log_entries = self._extract_span_event_logs(spans)
        hard_aligned_span_logs = align_logs_by_trace_id(span_log_entries, trace_id=trace_id)

        raw_logs = self._load_scenario_logs(scenario_name)
        hard_aligned_raw_logs = align_logs_by_trace_id(raw_logs, trace_id=trace_id)

        merged_entries: List[Dict[str, Any]] = []
        merged_entries.extend(hard_aligned_span_logs)
        merged_entries.extend(hard_aligned_raw_logs)

        # Fallback: if no explicit trace_id logs, align raw logs by time window + involved services.
        if not merged_entries:
            services = {str(s.get("service_code", "")) for s in spans if s.get("service_code")}
            for item in raw_logs:
                if services and item.get("service") not in services:
                    continue
                ts = float(item.get("timestamp", 0.0))
                if anchor_timestamp_s > 0 and abs(ts - anchor_timestamp_s) > self.log_window_seconds:
                    continue
                merged_entries.append(item)
            merged_entries.sort(key=lambda x: x.get("timestamp", 0.0))

        log_texts = [str(e.get("message", "")) for e in merged_entries if e.get("message")]

        if not log_texts:
            # Last fallback from trace span metadata (still from real trace data).
            log_texts = self._extract_log_texts(spans)

        return log_texts[: self.max_logs_per_trace]

    def _build_aligned_metrics(
        self,
        scenario_name: str,
        anchor_timestamp_s: float,
    ) -> Tuple[Tensor, Tensor]:
        """Build trace-aligned metric tensor from real metric CSV."""
        metric_series = self._load_scenario_metrics(scenario_name)

        if anchor_timestamp_s <= 0:
            # If trace summary timestamp is missing, fallback to the center of available metric timeline.
            non_empty = [m for m in metric_series if len(m.timestamps) > 0]
            if non_empty:
                first_ts = non_empty[0].timestamps
                anchor_timestamp_s = float(first_ts[len(first_ts) // 2])
            else:
                anchor_timestamp_s = 0.0

        timeline = np.linspace(
            anchor_timestamp_s - self.metric_window_seconds,
            anchor_timestamp_s + self.metric_window_seconds,
            self.max_metric_points,
            dtype=np.float64,
        )

        aligned_series = align_metrics_by_time_window(
            metrics=metric_series,
            center_timestamp=anchor_timestamp_s,
            window_seconds=self.metric_window_seconds,
        )

        metric_values = np.zeros((self.num_metrics, self.max_metric_points), dtype=np.float32)
        for i, series in enumerate(aligned_series[: self.num_metrics]):
            if len(series.timestamps) == 0:
                continue

            ts = series.timestamps.astype(np.float64)
            vals = series.values.astype(np.float32)
            order = np.argsort(ts)
            ts = ts[order]
            vals = vals[order]

            unique_ts, unique_idx = np.unique(ts, return_index=True)
            unique_vals = vals[unique_idx]

            if len(unique_ts) == 1:
                metric_values[i, :] = float(unique_vals[0])
            else:
                metric_values[i, :] = np.interp(
                    timeline,
                    unique_ts,
                    unique_vals,
                    left=float(unique_vals[0]),
                    right=float(unique_vals[-1]),
                ).astype(np.float32)

        relative_positions = compute_relative_time_positions(
            metric_timestamps=timeline,
            anchor_timestamp=anchor_timestamp_s,
        ).astype(np.float32)

        return (
            torch.from_numpy(metric_values),
            torch.from_numpy(relative_positions),
        )

    def _tokenize_logs(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        """Tokenize log texts into token IDs and attention masks."""
        if not texts:
            texts = ["[empty]"]

        tok = self.tokenizer
        if tok != "fallback":
            encoded = tok(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.log_max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"], encoded["attention_mask"]

        # Fallback: simple hash-based tokenization
        num_logs = len(texts)
        tokens = torch.zeros(num_logs, self.log_max_length, dtype=torch.long)
        masks = torch.zeros(num_logs, self.log_max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.split()[: self.log_max_length]
            for j, word in enumerate(words):
                tokens[i, j] = abs(hash(word)) % 30000 + 1
                masks[i, j] = 1
        return tokens, masks

# ----------------------------------------------------------------------
# Batch collation
# ----------------------------------------------------------------------


def collate_multimodal_samples(samples: List[MultimodalSample]) -> Dict:
    """Collate a list of MultimodalSample into a padded batch dict.

    Returns a dict with tensors suitable for the encoder/projector pipeline:
        span_features:           [B, max_spans, span_dim]
        trace_adjacency:         [B, max_spans, max_spans]
        log_tokens:              [B, max_logs, seq_len]
        log_mask:                [B, max_logs, seq_len]
        metric_values:           [B, num_metrics, metric_seq_len]
        relative_time_positions: [B, metric_seq_len]
        trace_ids:               list[str]
        layer:                   list[Optional[str]]
        labels:                  [B]
        fault_layers:            [B]
    """
    batch_size = len(samples)

    max_spans = max(s.span_features.shape[0] for s in samples)
    span_dim = samples[0].span_features.shape[1]
    max_logs = max(s.log_tokens.shape[0] for s in samples)
    log_seq_len = samples[0].log_tokens.shape[1]
    num_metrics = samples[0].metric_values.shape[0]
    metric_seq_len = max(s.metric_values.shape[1] for s in samples)

    # Pre-allocate padded tensors
    span_features = torch.zeros(batch_size, max_spans, span_dim)
    trace_adjacency = torch.zeros(batch_size, max_spans, max_spans)
    log_tokens = torch.zeros(batch_size, max_logs, log_seq_len, dtype=torch.long)
    log_mask = torch.zeros(batch_size, max_logs, log_seq_len, dtype=torch.long)
    metric_values = torch.zeros(batch_size, num_metrics, metric_seq_len)
    relative_time_positions = torch.zeros(batch_size, metric_seq_len)

    # Rich span features are present iff at least one sample carries them.
    # All-None batches behave byte-identical to pre-modification (no extra keys emitted).
    have_rich_spans = any(s.span_service_ids is not None for s in samples)
    if have_rich_spans:
        span_service_ids = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_operation_ids = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_status_classes = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_is_root = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_is_leaf = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_is_on_critical_path = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_error_flag = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_error_type_ids = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_otel_status_classes = torch.zeros(batch_size, max_spans, dtype=torch.long)

    trace_ids: List[str] = []
    layers: List[Optional[str]] = []
    labels: List[int] = []
    fault_layers: List[int] = []
    fault_type_ids: List[int] = []

    for i, s in enumerate(samples):
        ns = s.span_features.shape[0]
        span_features[i, :ns, :] = s.span_features
        trace_adjacency[i, :ns, :ns] = s.trace_adjacency

        nl = s.log_tokens.shape[0]
        log_tokens[i, :nl, :] = s.log_tokens
        log_mask[i, :nl, :] = s.log_attention_mask

        ms = s.metric_values.shape[1]
        metric_values[i, :, :ms] = s.metric_values
        rp_len = min(ms, s.relative_time_positions.shape[0])
        relative_time_positions[i, :rp_len] = s.relative_time_positions[:rp_len]

        if have_rich_spans and s.span_service_ids is not None:
            ks = min(ns, s.span_service_ids.shape[0])
            span_service_ids[i, :ks] = s.span_service_ids[:ks]
            span_operation_ids[i, :ks] = s.span_operation_ids[:ks]
            span_status_classes[i, :ks] = s.span_status_classes[:ks]
            span_is_root[i, :ks] = s.span_is_root[:ks]
            span_is_leaf[i, :ks] = s.span_is_leaf[:ks]
            if s.span_is_on_critical_path is not None:
                kc = min(ns, s.span_is_on_critical_path.shape[0])
                span_is_on_critical_path[i, :kc] = s.span_is_on_critical_path[:kc]
            if s.span_error_flag is not None:
                ke = min(ns, s.span_error_flag.shape[0])
                span_error_flag[i, :ke] = s.span_error_flag[:ke]
                span_error_type_ids[i, :ke] = s.span_error_type_ids[:ke]
                span_otel_status_classes[i, :ke] = s.span_otel_status_classes[:ke]

        trace_ids.append(s.trace_id)
        layers.append(s.layer)
        labels.append(s.label)
        fault_layers.append(s.fault_layer)
        fault_type_ids.append(s.fault_type_id)

    out: Dict[str, Any] = {
        "trace_ids": trace_ids,
        "span_features": span_features,
        "trace_adjacency": trace_adjacency,
        "log_tokens": log_tokens,
        "log_mask": log_mask,
        "metric_values": metric_values,
        "relative_time_positions": relative_time_positions,
        "layer": layers,
        "labels": torch.tensor(labels, dtype=torch.long),
        "fault_layers": torch.tensor(fault_layers, dtype=torch.long),
        "fault_type_ids": torch.tensor(fault_type_ids, dtype=torch.long),
    }
    if have_rich_spans:
        out["span_service_ids"] = span_service_ids
        out["span_operation_ids"] = span_operation_ids
        out["span_status_classes"] = span_status_classes
        out["span_is_root"] = span_is_root
        out["span_is_leaf"] = span_is_leaf
        out["span_is_on_critical_path"] = span_is_on_critical_path
        out["span_error_flag"] = span_error_flag
        out["span_error_type_ids"] = span_error_type_ids
        out["span_otel_status_classes"] = span_otel_status_classes
    return out
