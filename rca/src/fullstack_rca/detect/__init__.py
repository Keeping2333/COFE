"""Baseline-relative evidence detector.

Scores each telemetry element (metric channel / span by service+operation /
log pseudo-template) by how much the fault window deviates from the same
incident's own pre-fault normal window. The reference is always the same
service's own normal behavior, never a learned fault-type signature, so the
scoring stays robust on out-of-distribution fault types.

Public surface:
    parse_windows(meta)            -> FaultWindows (windows.py)
    EvidenceDetector.detect(...)   -> (metrics_summary, logs_summary, trace_summary)
"""

from .windows import FaultWindows, parse_windows
from .evidence_detector import EvidenceDetector

__all__ = [
    "FaultWindows",
    "parse_windows",
    "EvidenceDetector",
]
