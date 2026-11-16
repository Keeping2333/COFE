"""
Data loading and alignment utilities for Full-stack RCA.

Supports multiple dataset formats via config-driven factory:
- AnoMod (SkyWalking traces, pod logs, Prometheus CSV metrics)
- exp_runs (Jaeger traces, Loki logs, Prometheus JSON metrics)
- SN (SocialNetwork / Jaeger traces, text logs, CSV metrics)
"""

from .alignment import (
    MetricTimeSeries,
    align_logs_by_trace_id,
    align_metrics_by_time_window,
    build_trace_graph,
    compute_relative_time_positions,
    preprocess_jaeger_spans,
)
from .dataset import MicroServiceDataset, MultimodalSample, collate_multimodal_samples
from .exp_runs_dataset import ExpRunsDataset
from .factory import create_dataset, load_config
from .sn_dataset import SNDataset

__all__ = [
    # Datasets
    "MicroServiceDataset",
    "ExpRunsDataset",
    "SNDataset",
    # Factory
    "create_dataset",
    "load_config",
    # Core types
    "MultimodalSample",
    "collate_multimodal_samples",
    # Alignment utilities
    "MetricTimeSeries",
    "align_logs_by_trace_id",
    "align_metrics_by_time_window",
    "build_trace_graph",
    "compute_relative_time_positions",
    "preprocess_jaeger_spans",
]
