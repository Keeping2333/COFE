"""Fault / baseline window slicing for the evidence detector.

Every round3 ``meta.json`` carries an ISO-8601 UTC timestamp block::

    locust_start < inject_start < inject_end == recover_start < locust_end

The baseline (normal) window is the pre-fault span ``[locust_start, inject_start)``
(~30s, ``anomaly.fault_offset_sec``). The fault window is
``[inject_start, inject_end]`` (the 120s injection, ``anomaly.fault_duration_sec``).
The post-recovery span is deliberately NOT used as a normal reference: full
recovery is not guaranteed (connection-pool / JVM-GC / pod cold-start residue),
so it is not a clean baseline.

All telemetry timestamps live on the same epoch-seconds clock as the meta
block, so once the ISO strings are parsed to epoch floats, slicing is a numeric
mask:

  - metric series : Prometheus sample ts (float seconds)
  - log records   : ``timeUnixNano / 1e9``
  - trace spans   : ``startTime`` (microseconds) ``/ 1e6``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def iso_to_epoch(s: str) -> Optional[float]:
    """Parse an ISO-8601 UTC string (``2026-05-27T04:47:11Z``) to epoch seconds.

    Returns None when the string is missing or unparseable so callers can fall
    back rather than crash on a malformed meta block.
    """
    if not s:
        return None
    try:
        return datetime.strptime(s, _ISO_FMT).replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return None


@dataclass
class FaultWindows:
    """Epoch-second boundaries of an incident's baseline and fault windows,
    plus the ground-truth pointers the pointer-precision metric scores against.
    """

    baseline_start: float
    baseline_end: float          # == fault_start (inject_start)
    fault_start: float
    fault_end: float
    # Ground-truth pointers (from meta.anomaly); target_service is None for
    # node-level L1/L3 faults that are not scoped to a single service.
    target_service: Optional[str]
    layer: Optional[str]
    root_cause_type: Optional[str]
    run_id: Optional[str]

    @property
    def baseline_duration(self) -> float:
        return max(0.0, self.baseline_end - self.baseline_start)

    @property
    def fault_duration(self) -> float:
        return max(0.0, self.fault_end - self.fault_start)

    def baseline_mask(self, ts: np.ndarray) -> np.ndarray:
        """Boolean mask selecting samples in ``[baseline_start, baseline_end)``."""
        ts = np.asarray(ts, dtype=np.float64)
        return (ts >= self.baseline_start) & (ts < self.baseline_end)

    def fault_mask(self, ts: np.ndarray) -> np.ndarray:
        """Boolean mask selecting samples in ``[fault_start, fault_end]``."""
        ts = np.asarray(ts, dtype=np.float64)
        return (ts >= self.fault_start) & (ts <= self.fault_end)

    def in_fault(self, ts: float) -> bool:
        return self.fault_start <= float(ts) <= self.fault_end


def parse_windows(meta: dict) -> FaultWindows:
    """Build :class:`FaultWindows` from a round3 ``meta.json`` dict.

    Raises ValueError if the timestamp block lacks the inject boundaries, which
    are the minimum needed to define a fault window.
    """
    ts = (meta or {}).get("timestamps", {}) or {}
    anomaly = (meta or {}).get("anomaly", {}) or {}

    inject_start = iso_to_epoch(ts.get("inject_start"))
    inject_end = iso_to_epoch(ts.get("inject_end"))
    locust_start = iso_to_epoch(ts.get("locust_start"))

    if inject_start is None or inject_end is None:
        raise ValueError(
            f"meta.timestamps missing inject_start/inject_end "
            f"(run_id={meta.get('run_id')!r})"
        )

    # Baseline = pre-fault. If locust_start is missing or not before inject,
    # fall back to a fault-length window immediately before inject_start so the
    # baseline is never empty.
    if locust_start is None or locust_start >= inject_start:
        locust_start = inject_start - max(1.0, inject_end - inject_start)

    target = anomaly.get("target_service")
    # Some collectors record an empty string for node-level faults.
    if not target:
        target = None

    return FaultWindows(
        baseline_start=locust_start,
        baseline_end=inject_start,
        fault_start=inject_start,
        fault_end=inject_end,
        target_service=target,
        layer=anomaly.get("layer") or meta.get("root_cause_layer"),
        root_cause_type=anomaly.get("root_cause_type") or meta.get("root_cause_type"),
        run_id=meta.get("run_id"),
    )
