"""
Full-stack RCA: Multimodal Root Cause Analysis in Microservices.

This package implements a two-stage RCA system:
1. Fast Detection: Multimodal contrastive learning for anomaly detection
2. Deep Diagnosis: LLM-based reasoning with SOP validation

Modules:
- data: Dataset loading with trace-centric alignment
- models: Encoders, fusion, and contrastive losses
- rag: Retrieval-augmented generation for historical faults
- agent: LLM reasoning agents with SOP compliance
"""

__version__ = "0.1.0"

from .models.base import FaultLayer

__all__ = [
    "FaultLayer",
    "__version__",
]
