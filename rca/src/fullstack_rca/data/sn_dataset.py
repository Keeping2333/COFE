'English text removed.'
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .alignment import (
    MetricTimeSeries,
    align_metrics_by_time_window,
    build_trace_graph,
    compute_relative_time_positions,
    preprocess_jaeger_spans,
)
from .dataset import MultimodalSample

logger = logging.getLogger(__name__)

                    

_SN_LABEL_MAP = [
    ("Normal",       None,  -1, "normal"),
    ("Perf_CPU",     "L1",   0, "cpu_contention"),
    ("Perf_Disk",    "L1",   0, "disk_io_stress"),
    ("Perf_Network", "L2",   1, "network_loss"),
    ("DB_Redis",     "L3",   2, "redis_cache_limit"),
    ("Svc_Kill",     "L3",   2, "service_killed"),
    ("Code_Stop",    "L4",   3, "code_stop"),
]

                                         
_METRIC_FILES = [
    ("system_cpu_usage.csv",            None,    "sys_cpu"),
    ("system_memory_usage_percent.csv", None,    "sys_mem"),
    ("system_load1.csv",                None,    "sys_load1"),
    ("system_disk_io_time.csv",         None,    "sys_disk_io"),
    ("system_disk_read_bytes.csv",      None,    "sys_disk_read"),
    ("system_disk_write_bytes.csv",     None,    "sys_disk_write"),
    ("system_network_receive_bytes.csv",  "eth0", "sys_net_rx"),
    ("system_network_transmit_bytes.csv", "eth0", "sys_net_tx"),
    ("system_network_errors.csv",       None,    "sys_net_err"),
    ("socialnet_container_cpu.csv",     None,    "ctr_cpu_avg"),
]

                                               
_LOG_TS_RE = re.compile(
    r"^\[(\d{4}-\w{3}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\]"
)
_LOG_TS_FMT = "%Y-%b-%d %H:%M:%S.%f"


def _parse_sn_label(experiment_id: str) -> Tuple[Optional[str], int, str]:
    'English text removed.'
    for prefix, layer, layer_idx, rca in _SN_LABEL_MAP:
        if experiment_id.startswith(prefix):
            return layer, layer_idx, rca
    return None, -1, "unknown"


class SNDataset(Dataset):
    'English text removed.'

    def __init__(
        self,
        data_dir: str,
        metric_window_seconds: float = 60.0,
        log_window_seconds: float = 60.0,
        max_logs_per_trace: int = 20,
        num_metrics: int = 10,
        max_metric_points: int = 60,
        max_traces_per_scenario: int = 50,
        max_spans_per_trace: int = 30,
        log_max_length: int = 64,
        tokenizer_name: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metric_window_seconds = metric_window_seconds
        self.log_window_seconds = log_window_seconds
        self.max_logs_per_trace = max_logs_per_trace
        self.num_metrics = num_metrics
        self.max_metric_points = max_metric_points
        self.max_traces_per_scenario = max_traces_per_scenario
        self.max_spans_per_trace = max_spans_per_trace
        self.log_max_length = log_max_length
        self._tokenizer_name = tokenizer_name
        self._tokenizer = None

              
        self._samples_index: List[Dict] = []
            
        self._trace_cache: Dict[str, List[Dict]] = {}
        self._log_cache:    Dict[str, List[Dict]] = {}
        self._metric_cache: Dict[str, List[MetricTimeSeries]] = {}

        self._load_scenarios()

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

                      

    def _load_scenarios(self):
        'English text removed.'
        trace_root = self.data_dir / "trace_data"
        if not trace_root.exists():
            raise FileNotFoundError(f"trace_data/ not found under {self.data_dir}")

        scenario_dirs = sorted(d for d in trace_root.iterdir() if d.is_dir())
        logger.info("Found %d scenario dirs in %s", len(scenario_dirs), trace_root)

        for trace_dir in scenario_dirs:
            folder_name = trace_dir.name
                                                                
            if "_traces_" in folder_name:
                experiment_id = folder_name.split("_traces_")[0]
            else:
                experiment_id = folder_name

            layer, layer_idx, rca = _parse_sn_label(experiment_id)

                            
            trace_file = trace_dir / "all_traces.json"
            if not trace_file.exists():
                logger.warning("all_traces.json not found in %s, skipping", trace_dir)
                continue

            import json
            try:
                with open(trace_file) as f:
                    raw = json.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", trace_file, e)
                continue

                                        
            traces = raw.get("data", raw.get("traces", []))
            traces = [t for t in traces if t.get("spans")]             
            traces = traces[: self.max_traces_per_scenario]

            self._trace_cache[experiment_id] = traces
            logger.info(
                "  %s: %d traces (layer=%s, rca=%s)",
                experiment_id, len(traces), layer, rca,
            )

            for i, trace in enumerate(traces):
                trace_id = trace.get("traceID", f"{experiment_id}_{i}")
                                 
                start_us = min(
                    (s.get("startTime", 0) for s in trace.get("spans", [])),
                    default=0,
                )
                self._samples_index.append({
                    "experiment_id":   experiment_id,
                    "trace_idx":       i,
                    "trace_id":        trace_id,
                    "layer":           layer,
                    "layer_idx":       layer_idx,
                    "root_cause":      rca,
                    "start_timestamp_s": start_us / 1e6,
                })

        logger.info("Total SN samples indexed: %d", len(self._samples_index))

                            

    def __len__(self) -> int:
        return len(self._samples_index)

    def __getitem__(self, idx: int) -> MultimodalSample:
        info = self._samples_index[idx]
        exp_id = info["experiment_id"]
        trace = self._trace_cache[exp_id][info["trace_idx"]]

                                                   
        raw_spans = trace.get("spans", [])[: self.max_spans_per_trace]
        spans = preprocess_jaeger_spans(raw_spans)

        if spans:
            adjacency, span_features, _ = build_trace_graph(spans)
        else:
            adjacency = torch.zeros(1, 1)
            span_features = torch.zeros(1, 6)

        anchor_ts = info["start_timestamp_s"]

            
        log_texts = self._collect_logs(exp_id, anchor_ts)
        log_tokens, log_mask = self._tokenize_logs(log_texts)

        # Metrics
        metric_values, rel_positions = self._build_metrics(exp_id, anchor_ts)

        return MultimodalSample(
            trace_id=info["trace_id"],
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
        )

                      

    def _find_sibling_dir(self, experiment_id: str, modality: str) -> Optional[Path]:
        'English text removed.'
        base = self.data_dir / f"{modality}_data"
        if not base.exists():
            return None
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith(experiment_id):
                return d
        return None

    def _load_scenario_logs(self, experiment_id: str) -> List[Dict]:
        'English text removed.'
        if experiment_id in self._log_cache:
            return self._log_cache[experiment_id]

        entries: List[Dict] = []
        log_dir = self._find_sibling_dir(experiment_id, "log")
        if log_dir is None:
            self._log_cache[experiment_id] = entries
            return entries

        for log_file in sorted(log_dir.glob("*.log")):
            service = log_file.stem.rstrip("_")
            current_ts: Optional[float] = None
            current_lines: List[str] = []

            def flush(ts, lines):
                if ts is None or not lines:
                    return
                entries.append({
                    "timestamp": ts,
                    "service": service,
                    "message": " ".join(lines).strip(),
                })

            try:
                with open(log_file, errors="ignore") as f:
                    for raw_line in f:
                        line = raw_line.rstrip("\n")
                        m = _LOG_TS_RE.match(line)
                        if m:
                            flush(current_ts, current_lines)
                            try:
                                dt = datetime.strptime(m.group(1), _LOG_TS_FMT)
                                current_ts = dt.timestamp()
                            except ValueError:
                                current_ts = None
                            current_lines = [line]
                        elif current_ts is not None:
                            current_lines.append(line)
                    flush(current_ts, current_lines)
            except Exception as e:
                logger.warning("Failed to read log %s: %s", log_file, e)

        entries.sort(key=lambda x: x["timestamp"])
        self._log_cache[experiment_id] = entries
        return entries

    def _collect_logs(self, experiment_id: str, anchor_ts: float) -> List[str]:
        'English text removed.'
        all_logs = self._load_scenario_logs(experiment_id)

        if anchor_ts > 0:
            windowed = [
                e for e in all_logs
                if abs(e["timestamp"] - anchor_ts) <= self.log_window_seconds
            ]
        else:
            windowed = all_logs

        texts = [e["message"] for e in windowed if e.get("message")]
        return texts[: self.max_logs_per_trace] if texts else ["[no logs]"]

                            

    def _load_scenario_metrics(self, experiment_id: str) -> List[MetricTimeSeries]:
        'English text removed.'
        if experiment_id in self._metric_cache:
            return self._metric_cache[experiment_id]

        series_list: List[MetricTimeSeries] = []
        metric_dir = self._find_sibling_dir(experiment_id, "metric")

        if metric_dir:
            for csv_fname, filter_val, metric_name in _METRIC_FILES:
                csv_path = metric_dir / csv_fname
                if not csv_path.exists():
                    series_list.append(MetricTimeSeries(
                        name=metric_name,
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float32),
                    ))
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).map(
                        lambda x: x.timestamp()
                    )
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna(subset=["timestamp", "value"])

                                                           
                    if filter_val is not None:
                        for col in ("device", "container_label_com_docker_compose_service",
                                    "result", "instance"):
                            if col in df.columns:
                                df = df[df[col] == filter_val]
                                break

                                           
                    agg = df.groupby("timestamp")["value"].mean().reset_index()
                    agg = agg.sort_values("timestamp")
                    series_list.append(MetricTimeSeries(
                        name=metric_name,
                        timestamps=agg["timestamp"].to_numpy(dtype=np.float64),
                        values=agg["value"].to_numpy(dtype=np.float32),
                    ))
                except Exception as e:
                    logger.warning("Failed to load metric %s: %s", csv_fname, e)
                    series_list.append(MetricTimeSeries(
                        name=metric_name,
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float32),
                    ))

                                   
        while len(series_list) < self.num_metrics:
            series_list.append(MetricTimeSeries(
                name=f"missing_{len(series_list)}",
                timestamps=np.array([], dtype=np.float64),
                values=np.array([], dtype=np.float32),
            ))

        self._metric_cache[experiment_id] = series_list[:self.num_metrics]
        return self._metric_cache[experiment_id]

    def _build_metrics(
        self,
        experiment_id: str,
        anchor_ts: float,
    ) -> Tuple[Tensor, Tensor]:
        'English text removed.'
        metric_series = self._load_scenario_metrics(experiment_id)

                                       
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

        rel_pos = compute_relative_time_positions(timeline, anchor_ts).astype(np.float32)
        return torch.from_numpy(metric_values), torch.from_numpy(rel_pos)

    # ===== Log Tokenization =====

    def _tokenize_logs(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
        'English text removed.'
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

        # Fallback
        num_logs = len(texts)
        tokens = torch.zeros(num_logs, self.log_max_length, dtype=torch.long)
        masks  = torch.zeros(num_logs, self.log_max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.split()[: self.log_max_length]
            for j, word in enumerate(words):
                tokens[i, j] = abs(hash(word)) % 30000 + 1
                masks[i, j] = 1
        return tokens, masks

                      

    def get_stats(self) -> Dict:
        from collections import Counter
        layers = [info["layer"] or "Normal" for info in self._samples_index]
        rcas   = [info["root_cause"] for info in self._samples_index]
        return {
            "total": len(self._samples_index),
            "by_layer": dict(Counter(layers)),
            "by_rca":   dict(Counter(rcas)),
        }
