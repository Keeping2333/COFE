'English text removed.'
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .alignment import (
    MetricTimeSeries,
    align_metrics_by_time_window,
    build_trace_graph,
    compute_relative_time_positions,
    extract_rich_span_features,
    preprocess_jaeger_spans,
)
from .dataset import MultimodalSample, get_fault_type_id

logger = logging.getLogger(__name__)


_LOG_NORMALIZATION_PATTERNS = (
    # Order matters: mask structured identifiers before generic numbers.
    (re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE), "urltoken"),
    (re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE), "emailtoken"),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}(?:[t\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:z|[+-]\d{2}:?\d{2})?)?\b",
            re.IGNORECASE,
        ),
        "timestamptoken",
    ),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "timetoken"),
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        "uuidtoken",
    ),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b"), "ipaddrtoken"),
    (re.compile(r"\b(?:[0-9a-f]{1,4}:){2,}[0-9a-f]{1,4}\b", re.IGNORECASE), "ipaddrtoken"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE), "hexidtoken"),
    (re.compile(r"\b\d{6,}\b"), "numtoken"),
)


def _normalize_log_for_encoder(text: str) -> str:
    """Lowercase, mask high-cardinality identifiers (IP / timestamp / trace id /
    long numbers), then strip every remaining special character so only letters,
    digits, and single spaces survive. Masking runs first so the placeholder
    tokens are plain words that the strip step preserves."""
    normalized = str(text or "").lower()
    for pattern, replacement in _LOG_NORMALIZATION_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "empty"


                    
_LAYER_TO_IDX = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}

# Round2 audit (2026-05-04) flagged these three experiments as having
# sparse / near-empty log streams that would distort both training and
# evaluation. They are filtered at index time so neither the train nor
# the test split ever sees them. Match by run_id prefix because the
# run_id includes the target service and timestamp suffix.
# Round2 sparse-log audit (P1, 2026-04-30) originally flagged these prefixes:
#   - L2_02_network_latency_sample04_
#   - L2_03_packet_loss_sample13_
#   - L4_02_database_connection_exhaustion_sample30_
# They are now KEPT in the dataset to preserve the full 360 round2 experiments.
# Repopulate this tuple if a future audit identifies new bad runs.
SPARSE_LOG_EXCLUDED_RUN_ID_PREFIXES: tuple[str, ...] = ()


def _is_sparse_log_excluded(run_id: str) -> bool:
    return any(run_id.startswith(p) for p in SPARSE_LOG_EXCLUDED_RUN_ID_PREFIXES)


def _otel_attr_to_dict(attrs: List[Dict]) -> Dict[str, Any]:
    """Flatten an OTel ``attributes`` array into a key→primitive dict.

    OTel encodes each attribute as ``{"key": str, "value": {"stringValue": ...,
    "intValue": ..., "boolValue": ..., ...}}``. We unwrap the first concrete
    value variant; if none is present the key is dropped silently.
    """
    out: Dict[str, Any] = {}
    for a in attrs or []:
        key = a.get("key")
        if not key:
            continue
        val_obj = a.get("value") or {}
        for vk in ("stringValue", "intValue", "boolValue", "doubleValue"):
            if vk in val_obj:
                out[key] = val_obj[vk]
                break
    return out

# Round3 ships 61 Prometheus query_range JSON files per experiment, covering
# all four fault layers plus CoreDNS (L2 DNS) and kube-state (L3 Platform)
# streams. Round3 adds system_disk_pending_operations (60→61). We expose all
# 61 to the encoder; the 1D-CNN + Gaussian-attention metric encoder learns
# per-channel weights so even sparse / mostly-zero series (e.g. resourcequota
# only fires under L3_03) contribute discriminative power across fault types.
_METRIC_FILES = [
    # --- L1 Infrastructure: system (12) ---
    "system_cpu_utilization.json",
    "system_cpu_by_mode.json",
    "system_cpu_load_1m.json",
    "system_cpu_load_5m.json",
    "system_memory_usage.json",
    "system_memory_utilization.json",
    "system_disk_io_bytes.json",
    "system_disk_operations.json",
    "system_disk_io_time.json",
    "system_disk_weighted_io_time.json",
    "system_disk_pending_operations.json",   # new in round3 (60→61 channels)
    "system_filesystem_usage.json",
    # --- L2 Network: NIC counters (5) ---
    "network_io_bytes.json",
    "network_packets.json",
    "network_errors.json",
    "network_dropped.json",
    "network_connections_by_state.json",
    # --- L2 DNS: CoreDNS (7, new in round2) ---
    "coredns_request_count.json",
    "coredns_request_duration_p99.json",
    "coredns_responses_total.json",
    "coredns_response_nxdomain.json",
    "coredns_response_servfail.json",
    "coredns_panics_total.json",
    "coredns_cache_hits.json",
    # --- L3 Platform: kube-state-metrics (9, new in round2) ---
    "kube_pod_phase.json",
    "kube_pod_ready.json",
    "kube_pod_restarts_total.json",
    "kube_pod_waiting_reason.json",
    "kube_deployment_status_replicas.json",
    "kube_deployment_status_replicas_available.json",
    "kube_deployment_status_replicas_unavailable.json",
    "kube_resourcequota_hard.json",
    "kube_resourcequota_used.json",
    # --- L3 Platform: scrape health (2) ---
    "service_up_status.json",
    "scrape_targets_health.json",
    # --- L4 Application: HTTP server + client (7) ---
    "http_request_rate.json",
    "http_request_latency_p99.json",
    "http_request_latency_p95.json",
    "http_request_latency_p50.json",
    "http_request_latency_avg.json",
    "http_client_request_rate.json",
    "http_client_latency_avg.json",
    # --- L4 Application: JVM (9) ---
    "jvm_memory_used.json",
    "jvm_memory_limit.json",
    "jvm_memory_committed.json",
    "jvm_gc_duration.json",
    "jvm_gc_count.json",
    "jvm_thread_count.json",
    "jvm_class_count.json",
    "jvm_cpu_utilization.json",
    "jvm_cpu_time.json",
    # --- L4 Application: DB connections / HikariCP (7) ---
    "db_connections_usage.json",
    "db_connections_max.json",
    "db_connections_idle_min.json",
    "db_connections_pending.json",
    "db_connections_wait_time.json",
    "db_connections_use_time.json",
    "db_connections_create_time.json",
    # --- Telemetry pipeline (3) ---
    "otel_processed_spans.json",
    "otel_processed_logs.json",
    "otel_queue_size.json",
]
assert len(_METRIC_FILES) == 61, f"expected 61 metric files, got {len(_METRIC_FILES)}"

# SocialNetwork (round_sn1) metric channels — 27 files, present and identical
# across all 360 experiments. Hand-ordered TT-style (host system / host NIC /
# container / otel pipeline). Selected when the dataset is created with
# system="sn".
_METRIC_FILES_SN = [
    # --- host system (10) ---
    "system_cpu_utilization.json",
    "system_cpu_by_mode.json",
    "system_cpu_load_1m.json",
    "system_cpu_load_5m.json",
    "system_memory_usage.json",
    "system_memory_utilization.json",
    "system_disk_io_bytes.json",
    "system_disk_io_time.json",
    "system_disk_operations.json",
    "system_filesystem_usage.json",
    # --- host NIC (5) ---
    "network_io_bytes.json",
    "network_packets.json",
    "network_errors.json",
    "network_dropped.json",
    "network_connections_by_state.json",
    # --- container (9) ---
    "container_cpu_utilization.json",
    "container_memory_usage.json",
    "container_memory_percent.json",
    "container_memory_limit.json",
    "container_blockio_bytes.json",
    "container_network_rx_bytes.json",
    "container_network_tx_bytes.json",
    "container_network_rx_dropped.json",
    "container_network_tx_dropped.json",
    # --- otel pipeline (3) ---
    "otel_received_spans.json",
    "otel_sent_spans.json",
    "otel_received_logs.json",
]
assert len(_METRIC_FILES_SN) == 27, f"expected 27 SN metric files, got {len(_METRIC_FILES_SN)}"

_METRIC_FILE_SETS = {"tt": _METRIC_FILES, "sn": _METRIC_FILES_SN}


class ExpRunsDataset(Dataset):
    'English text removed.'

    def __init__(
        self,
        data_dir: str,
        metric_window_seconds: float = 60.0,
        max_logs_per_trace: int = 20,
        num_metrics: int = 61,   # round3: 61 metric files per experiment
        max_metric_points: int = 60,
        max_traces_per_experiment: int = 200,
        max_spans_per_trace: int = 30,
        min_spans_per_trace: int = 2,
        log_max_length: int = 64,
        tokenizer_name: str = "distilbert-base-uncased",
        duration_stats: Optional[Dict[str, float]] = None,
        enable_rich_span_features: bool = False,
        num_service_buckets: int = 64,
        num_operation_buckets: int = 1024,
        num_error_type_buckets: int = 32,
        mask_metric_channels: Optional[List[str]] = None,
        system: str = "tt",
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.system = system
        self.metric_files = _METRIC_FILE_SETS.get(system, _METRIC_FILES)
        self.metric_window_seconds = metric_window_seconds
        self.max_logs_per_trace = max_logs_per_trace
        self.num_metrics = num_metrics
        self.max_metric_points = max_metric_points
        self.max_traces_per_experiment = max_traces_per_experiment
        self.max_spans_per_trace = max_spans_per_trace
        self.min_spans_per_trace = min_spans_per_trace
        self.log_max_length = log_max_length
        self._tokenizer_name = tokenizer_name
        self._tokenizer = None

        # Rich span features (T1+T2): off by default — when False, the
        # __getitem__ output and the resulting batch dict are byte-identical
        # to pre-modification.
        self.enable_rich_span_features = enable_rich_span_features
        self.num_service_buckets = num_service_buckets
        self.num_operation_buckets = num_operation_buckets
        self.num_error_type_buckets = num_error_type_buckets

        # Duration stats for log1p + z-score normalization (set by training script)
        self.duration_stats = duration_stats

        # Optional metric-channel masking (default OFF = identical behavior).
        # Resolve channel names to tensor row indices via _METRIC_FILES order;
        # _build_metrics zeroes these rows so they contribute nothing to the
        # encoder's first Conv1d and are auto-skipped by _summarize_metrics.
        self.mask_metric_channels = list(mask_metric_channels or [])
        _name_to_idx = {Path(f).stem: i for i, f in enumerate(self.metric_files)}
        self._mask_metric_indices = sorted(
            _name_to_idx[n] for n in self.mask_metric_channels if n in _name_to_idx
        )
        _unknown = [n for n in self.mask_metric_channels if n not in _name_to_idx]
        if _unknown:
            logger.warning("mask_metric_channels: unknown channel(s) ignored: %s", _unknown)
        if self._mask_metric_indices:
            logger.info(
                "Masking %d metric channel(s) (zeroed): %s",
                len(self._mask_metric_indices), self.mask_metric_channels,
            )

              
        self._samples_index: List[Dict] = []
                                                     
        self._trace_cache: Dict[str, List[Tuple[str, List[Dict]]]] = {}
                                          
        self._log_cache: Dict[str, List[Dict]] = {}
                                          
        self._metric_cache: Dict[str, List[MetricTimeSeries]] = {}

        self._load_experiments()

    # ===== Tokenizer =====

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            except Exception as e:
                logger.error(
                    "Failed to load tokenizer '%s' (%s) — falling back to hash "
                    "tokens, which are NON-DETERMINISTIC and INCOMPATIBLE with a KB "
                    "built using the real tokenizer; retrieval will be garbage. "
                    "Set HF_HUB_OFFLINE=1 to force the local cache.",
                    self._tokenizer_name, e,
                )
                self._tokenizer = "fallback"
        return self._tokenizer

                      

    def _load_experiments(self):
        'English text removed.'
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # round2 directory names are L1_/L2_/L3_/L4_-prefixed (e.g.
        # L1_01_node_cpu_pressure_sample01_20260428_123229), so we no longer
        # filter by name. Presence of meta.json is the canonical experiment marker.
        exp_dirs = sorted(
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )
        logger.info("Found %d experiment dirs in %s", len(exp_dirs), self.data_dir)

        for exp_dir in exp_dirs:
            meta_file = exp_dir / "meta.json"
            if not meta_file.exists():
                logger.warning("meta.json not found in %s, skipping", exp_dir)
                continue

            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", meta_file, e)
                continue

            run_id = meta.get("run_id", exp_dir.name)
            if _is_sparse_log_excluded(run_id):
                logger.info("Skipping sparse-log experiment %s (round2 audit excluded)", run_id)
                continue
            anomaly = meta.get("anomaly", {})
            layer = anomaly.get("layer")
            root_cause_type = anomaly.get("root_cause_type", "unknown")
            layer_idx = _LAYER_TO_IDX.get(layer, -1)

                           
            trace_dir = exp_dir / "traces"
            if not trace_dir.exists():
                logger.warning("traces/ not found in %s, skipping", exp_dir)
                continue

            trace_files = sorted(trace_dir.glob("*.json"))
            if not trace_files:
                logger.warning("No trace files in %s, skipping", trace_dir)
                continue

            # Quality-aware sampling: filter trivial traces, then cap.
            trace_files = self._quality_sample_traces(
                trace_files, exp_dir.name,
            )

                               
            timestamps = meta.get("timestamps", {})
            inject_start = timestamps.get("inject_start", "")
            inject_end = timestamps.get("inject_end", "")

                           
            inject_center_s = 0.0
            if inject_start and inject_end:
                try:
                    from datetime import datetime, timezone
                    t0 = datetime.fromisoformat(inject_start.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(inject_end.replace("Z", "+00:00"))
                    inject_center_s = (t0.timestamp() + t1.timestamp()) / 2.0
                except Exception:
                    pass

                                
            cached_traces: List[Tuple[str, List[Dict]]] = []
            for trace_file in trace_files:
                trace_id = trace_file.stem
                cached_traces.append((trace_id, trace_file))

            self._trace_cache[run_id] = cached_traces

            logger.info(
                "  %s: %d traces (layer=%s, rca=%s)",
                run_id, len(trace_files), layer, root_cause_type,
            )

            for i, (trace_id, _) in enumerate(cached_traces):
                self._samples_index.append({
                    "run_id": run_id,
                    "exp_dir": str(exp_dir),
                    "trace_idx": i,
                    "trace_id": trace_id,
                    "layer": layer,
                    "layer_idx": layer_idx,
                    "root_cause": root_cause_type,
                    "inject_center_s": inject_center_s,
                })

        logger.info("Total ExpRuns samples indexed: %d", len(self._samples_index))

    def _quality_sample_traces(
        self, trace_files: List[Path], exp_name: str,
    ) -> List[Path]:
        """Filter out trivial traces, then cap to max_traces_per_experiment.

        Strategy:
        1. Quick-scan each file to count spans (just parse top-level JSON).
        2. Keep only traces with >= min_spans_per_trace spans.
        3. Sort by span count descending (richest first).
        4. If still over limit, take the top ones + random sample from the rest
           to maintain some diversity.
        """
        if self.min_spans_per_trace <= 0 and len(trace_files) <= self.max_traces_per_experiment:
            return trace_files

        scored = []
        for tf in trace_files:
            try:
                with open(tf) as f:
                    raw = json.load(f)
                n_spans = len(raw.get("data", [{}])[0].get("spans", []))
            except Exception:
                n_spans = 0
            scored.append((n_spans, tf))

        total_before = len(scored)
        scored = [(n, tf) for n, tf in scored if n >= self.min_spans_per_trace]
        filtered_out = total_before - len(scored)

        if filtered_out > 0:
            logger.info(
                "  %s: filtered %d/%d trivial traces (<%d spans)",
                exp_name, filtered_out, total_before, self.min_spans_per_trace,
            )

        if not scored:
            logger.warning("  %s: all traces filtered out, keeping top 10 by span count", exp_name)
            scored = sorted(
                [(n, tf) for n, tf in [(n, tf) for tf in trace_files
                 for n in [len(json.load(open(tf)).get("data", [{}])[0].get("spans", []))]
                ]],
                key=lambda x: x[0], reverse=True,
            )[:10]

        if len(scored) <= self.max_traces_per_experiment:
            return sorted([tf for _, tf in scored])

        scored.sort(key=lambda x: x[0], reverse=True)
        top_half = self.max_traces_per_experiment // 2
        top_traces = scored[:top_half]
        rest = scored[top_half:]

        rng = random.Random(42)
        remaining_needed = self.max_traces_per_experiment - top_half
        sampled_rest = rng.sample(rest, min(remaining_needed, len(rest)))

        result = [tf for _, tf in top_traces] + [tf for _, tf in sampled_rest]
        return sorted(result)

    def _load_trace(self, run_id: str, trace_idx: int) -> Tuple[str, List[Dict]]:
        'English text removed.'
        trace_id, spans, _ = self._load_trace_with_processes(run_id, trace_idx)
        return trace_id, spans

    def _load_trace_with_processes(
        self, run_id: str, trace_idx: int,
    ) -> Tuple[str, List[Dict], Dict[str, Dict]]:
        """Load a trace and also return the trace-level ``processes`` map.

        The processes map is required to resolve ``processID`` -> ``serviceName``
        for rich span feature extraction. We expose it via a separate method
        so the legacy two-tuple signature stays intact.
        """
        entry = self._trace_cache[run_id][trace_idx]
        trace_id = entry[0]
        trace_file = entry[1]

                                                         
        if isinstance(trace_file, list):
                        
            return trace_id, trace_file, {}

        try:
            with open(trace_file) as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning("Failed to load trace %s: %s", trace_file, e)
            return trace_id, [], {}

                                                                                     
        data = raw.get("data", [])
        if data and data[0].get("spans"):
            spans = data[0]["spans"]
            processes = data[0].get("processes", {}) or {}
        else:
            spans = []
            processes = {}

                                               
        return trace_id, spans, processes

                            

    def __len__(self) -> int:
        return len(self._samples_index)

    def __getitem__(self, idx: int) -> MultimodalSample:
        info = self._samples_index[idx]
        run_id = info["run_id"]

                                                                       
        trace_id, raw_spans, processes_map = self._load_trace_with_processes(
            run_id, info["trace_idx"]
        )
        raw_spans = raw_spans[: self.max_spans_per_trace]

                                                   
        spans = preprocess_jaeger_spans(raw_spans)

        if spans:
            adjacency, span_features, _ = build_trace_graph(
                spans,
                duration_stats=self.duration_stats,
            )
        else:
            adjacency = torch.zeros(1, 1)
            span_features = torch.zeros(1, 6)

        # Rich span features (T1+T2+T3 + round2 error tags): only computed
        # when the dataset is constructed with enable_rich_span_features=True.
        # Feature-flag-off path leaves these as None and the collate_fn omits them.
        span_service_ids = None
        span_operation_ids = None
        span_status_classes = None
        span_is_root = None
        span_is_leaf = None
        span_is_on_critical_path = None
        span_error_flag = None
        span_error_type_ids = None
        span_otel_status_classes = None
        if self.enable_rich_span_features:
            if raw_spans:
                rich = extract_rich_span_features(
                    raw_spans,
                    processes_map=processes_map,
                    num_service_buckets=self.num_service_buckets,
                    num_operation_buckets=self.num_operation_buckets,
                    num_error_type_buckets=self.num_error_type_buckets,
                )
                span_service_ids = rich["service_ids"]
                span_operation_ids = rich["operation_ids"]
                span_status_classes = rich["status_classes"]
                span_is_root = rich["is_root"]
                span_is_leaf = rich["is_leaf"]
                span_is_on_critical_path = rich["is_on_critical_path"]
                span_error_flag = rich["error_flag"]
                span_error_type_ids = rich["error_type_ids"]
                span_otel_status_classes = rich["otel_status_classes"]
            else:
                # Match the empty-trace fallback above so num_spans=1 padding works.
                span_service_ids = torch.zeros(1, dtype=torch.long)
                span_operation_ids = torch.zeros(1, dtype=torch.long)
                span_status_classes = torch.zeros(1, dtype=torch.long)
                span_is_root = torch.zeros(1, dtype=torch.long)
                span_is_leaf = torch.zeros(1, dtype=torch.long)
                span_is_on_critical_path = torch.zeros(1, dtype=torch.long)
                span_error_flag = torch.zeros(1, dtype=torch.long)
                span_error_type_ids = torch.zeros(1, dtype=torch.long)
                span_otel_status_classes = torch.zeros(1, dtype=torch.long)

                                               
        anchor_ts = 0.0
        if raw_spans:
            start_us = min(
                (s.get("startTime", 0) for s in raw_spans),
                default=0,
            )
            anchor_ts = start_us / 1e6

            
        log_texts = self._collect_logs(info["exp_dir"], run_id, trace_id, anchor_ts)
        log_tokens, log_mask = self._tokenize_logs(log_texts)

                                    
        metric_anchor = info["inject_center_s"] if info["inject_center_s"] > 0 else anchor_ts
        metric_values, rel_positions = self._build_metrics(info["exp_dir"], run_id, metric_anchor)

        return MultimodalSample(
            trace_id=trace_id,
            span_features=span_features,
            trace_adjacency=adjacency,
            log_tokens=log_tokens,
            log_attention_mask=log_mask,
            metric_values=metric_values,
            relative_time_positions=rel_positions,
            label=max(0, info["layer_idx"]),
            fault_layer=info["layer_idx"],
            layer=info["layer"],
            root_cause_type=info["root_cause"],
            num_logs=len(log_texts),
            fault_type_id=get_fault_type_id(info["root_cause"], system=self.system),
            span_service_ids=span_service_ids,
            span_operation_ids=span_operation_ids,
            span_status_classes=span_status_classes,
            span_is_root=span_is_root,
            span_is_leaf=span_is_leaf,
            span_is_on_critical_path=span_is_on_critical_path,
            span_error_flag=span_error_flag,
            span_error_type_ids=span_error_type_ids,
            span_otel_status_classes=span_otel_status_classes,
        )

                      

    def _load_experiment_logs(self, exp_dir: str, run_id: str) -> List[Dict]:
        """Load and parse round2 OTel jsonl logs.

        Each line of ``logs/logs.jsonl`` is a batch JSON wrapper; real
        LogRecords live three levels deep at
        ``resourceLogs[i].scopeLogs[j].logRecords[k]``.

        trace_id is read from the canonical top-level ``LogRecord.traceId``
        field (camelCase, OTel OTLP convention). Coverage in round2 is 99.7%
        per ``train-ticket/analysis/round2_quality_report.md``. Do NOT read
        ``attributes[key=='trace_id'].stringValue`` — that snake_case mirror
        is only ~50% populated and would fabricate a false coverage gap.
        """
        if run_id in self._log_cache:
            return self._log_cache[run_id]

        entries: List[Dict] = []
        log_file = Path(exp_dir) / "logs" / "logs.jsonl"
        if not log_file.exists():
            self._log_cache[run_id] = entries
            return entries

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
                        res_attrs = _otel_attr_to_dict(
                            (rl.get("resource") or {}).get("attributes") or []
                        )
                        res_service = res_attrs.get("service.name")
                        for sl in rl.get("scopeLogs", []) or []:
                            for lr in sl.get("logRecords", []) or []:
                                ts_ns_raw = lr.get("timeUnixNano") or lr.get("observedTimeUnixNano")
                                try:
                                    ts_s = int(ts_ns_raw) / 1e9 if ts_ns_raw else 0.0
                                except (TypeError, ValueError):
                                    ts_s = 0.0
                                body = (lr.get("body") or {}).get("stringValue", "")
                                # Canonical top-level traceId (99.7% in round2).
                                # Empty string is treated as missing.
                                trace_id = lr.get("traceId") or None
                                if res_service:
                                    service = res_service
                                elif self.system == "sn":
                                    # SN (Docker Compose filelog): no resource-level
                                    # service.name. round_sn2 enriches each LogRecord
                                    # with a human-readable service.name / container.name;
                                    # round_sn1 carries only container_id / log.file.path.
                                    # Prefer the readable name, fall back to a 12-char
                                    # container id, then the container dir in the path.
                                    rec_attrs = _otel_attr_to_dict(lr.get("attributes") or [])
                                    service = (
                                        rec_attrs.get("service.name")
                                        or rec_attrs.get("container.name")
                                        or ""
                                    )
                                    if not service:
                                        cid = rec_attrs.get("container_id") or ""
                                        if not cid:
                                            segs = [s for s in str(rec_attrs.get("log.file.path", "")).split("/") if s]
                                            cid = segs[-2] if len(segs) >= 2 else ""
                                        service = str(cid)[:12] if cid else "unknown"
                                else:
                                    service = "unknown"
                                entries.append({
                                    "timestamp": ts_s,
                                    "service": service,
                                    "message": str(body),
                                    "traceid": trace_id,
                                })
        except Exception as e:
            logger.warning("Failed to load OTel jsonl %s: %s", log_file, e)

        entries.sort(key=lambda x: x.get("timestamp", 0))
        self._log_cache[run_id] = entries
        return entries

    def _collect_logs(
        self, exp_dir: str, run_id: str, trace_id: str, anchor_ts: float,
    ) -> List[str]:
        'English text removed.'
        all_logs = self._load_experiment_logs(exp_dir, run_id)

                          
        trace_id_lower = trace_id.lower()
        matched = [
            e for e in all_logs
            if e.get("traceid") and str(e["traceid"]).lower() == trace_id_lower
        ]

                   
        if not matched and anchor_ts > 0:
            matched = [
                e for e in all_logs
                if abs(e.get("timestamp", 0) - anchor_ts) <= self.metric_window_seconds
            ]

                   
        if not matched:
            matched = all_logs

        texts = [e["message"] for e in matched if e.get("message")]
        texts = texts[: self.max_logs_per_trace]
        return texts if texts else ["[no logs]"]

                            

    def _load_experiment_metrics(self, exp_dir: str, run_id: str) -> List[MetricTimeSeries]:
        'English text removed.'
        if run_id in self._metric_cache:
            return self._metric_cache[run_id]

        series_list: List[MetricTimeSeries] = []
        metric_dir = Path(exp_dir) / "metrics"

        if metric_dir.exists():
            for metric_fname in self.metric_files:
                metric_path = metric_dir / metric_fname
                if not metric_path.exists():
                    series_list.append(MetricTimeSeries(
                        name=metric_fname.replace(".json", ""),
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float32),
                    ))
                    continue

                try:
                    with open(metric_path) as f:
                        raw = json.load(f)
                except Exception as e:
                    logger.warning("Failed to load metric %s: %s", metric_path, e)
                    series_list.append(MetricTimeSeries(
                        name=metric_fname.replace(".json", ""),
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float32),
                    ))
                    continue

                result = raw.get("data", {}).get("result", [])
                if not result:
                    series_list.append(MetricTimeSeries(
                        name=metric_fname.replace(".json", ""),
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float32),
                    ))
                    continue

                                              
                all_ts = []
                all_vals = []
                for r in result:
                    for ts_val, val_str in r.get("values", []):
                        try:
                            all_ts.append(float(ts_val))
                            all_vals.append(float(val_str))
                        except (ValueError, TypeError):
                            continue

                if all_ts:
                    ts_arr = np.array(all_ts, dtype=np.float64)
                    val_arr = np.array(all_vals, dtype=np.float32)
                                
                    unique_ts = np.unique(ts_arr)
                    agg_vals = np.array([
                        val_arr[ts_arr == t].mean() for t in unique_ts
                    ], dtype=np.float32)
                    order = np.argsort(unique_ts)
                    series_list.append(MetricTimeSeries(
                        name=metric_fname.replace(".json", ""),
                        timestamps=unique_ts[order],
                        values=agg_vals[order],
                    ))
                else:
                    series_list.append(MetricTimeSeries(
                        name=metric_fname.replace(".json", ""),
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float32),
                    ))

                                   
        while len(series_list) < self.num_metrics:
            series_list.append(MetricTimeSeries(
                name=f"missing_{len(series_list)}",
                timestamps=np.array([], dtype=np.float64),
                values=np.array([], dtype=np.float32),
            ))

        self._metric_cache[run_id] = series_list[: self.num_metrics]
        return self._metric_cache[run_id]

    def _build_metrics(
        self, exp_dir: str, run_id: str, anchor_ts: float,
    ) -> Tuple[Tensor, Tensor]:
        'English text removed.'
        metric_series = self._load_experiment_metrics(exp_dir, run_id)

        if anchor_ts <= 0:
            non_empty = [m for m in metric_series if len(m.timestamps) > 0]
            if non_empty:
                ts = non_empty[0].timestamps
                anchor_ts = float(ts[len(ts) // 2])
            else:
                anchor_ts = 0.0

        timeline = np.linspace(
            anchor_ts - self.metric_window_seconds,
            anchor_ts + self.metric_window_seconds,
            self.max_metric_points,
            dtype=np.float64,
        )

        aligned = align_metrics_by_time_window(
            metrics=metric_series,
            center_timestamp=anchor_ts,
            window_seconds=self.metric_window_seconds,
        )

        metric_values = np.zeros((self.num_metrics, self.max_metric_points), dtype=np.float32)
        for i, series in enumerate(aligned[: self.num_metrics]):
            if len(series.timestamps) == 0:
                continue
            ts = series.timestamps.astype(np.float64)
            vals = series.values.astype(np.float32)
            # Drop NaN/Inf values before interpolation (Prometheus
            # returns NaN for metrics with no observations in a window)
            valid = np.isfinite(vals)
            ts, vals = ts[valid], vals[valid]
            if len(ts) == 0:
                continue
            order = np.argsort(ts)
            ts, vals = ts[order], vals[order]
            unique_ts, unique_idx = np.unique(ts, return_index=True)
            unique_vals = vals[unique_idx]
            if len(unique_ts) == 1:
                metric_values[i, :] = float(unique_vals[0])
            else:
                metric_values[i, :] = np.interp(
                    timeline, unique_ts, unique_vals,
                    left=float(unique_vals[0]),
                    right=float(unique_vals[-1]),
                ).astype(np.float32)

        # Zero out masked channels (default: none → no-op, identical behavior).
        for mi in self._mask_metric_indices:
            if mi < self.num_metrics:
                metric_values[mi, :] = 0.0

        rel_pos = compute_relative_time_positions(timeline, anchor_ts).astype(np.float32)
        return torch.from_numpy(metric_values), torch.from_numpy(rel_pos)

    # ===== Log Tokenization =====

    def _tokenize_logs(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        'English text removed.'
        if not texts:
            texts = ["[empty]"]
        texts = [_normalize_log_for_encoder(text) for text in texts]

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

        # Fallback
        num_logs = len(texts)
        tokens = torch.zeros(num_logs, self.log_max_length, dtype=torch.long)
        masks = torch.zeros(num_logs, self.log_max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.split()[: self.log_max_length]
            for j, word in enumerate(words):
                tokens[i, j] = abs(hash(word)) % 30000 + 1
                masks[i, j] = 1
        return tokens, masks

                      

    def get_stats(self) -> Dict:
        from collections import Counter
        layers = [info["layer"] or "Normal" for info in self._samples_index]
        rcas = [info["root_cause"] for info in self._samples_index]
        return {
            "total": len(self._samples_index),
            "by_layer": dict(Counter(layers)),
            "by_rca": dict(Counter(rcas)),
        }
