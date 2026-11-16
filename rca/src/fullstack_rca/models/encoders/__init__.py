"""
Encoders module for Full-stack RCA.

Provides modality-specific encoders:
- LogEncoder: DistilBERT for log encoding
- MetricEncoder: 1D-CNN + Gaussian Attention for metric encoding
- TraceEncoder: Graph Attention Network for trace encoding
"""
from .gaussian_attention import (
    GaussianAttention,
    GaussianPositionalEncoding,
    MetricEncoderWithGaussianAttention,
    create_metric_encoder,
)
from .log_encoder import LogEncoder, SimplifiedLogEncoder
from .trace_encoder import SimpleTraceEncoder, TraceEncoder

__all__ = [
    # Gaussian Attention
    "GaussianPositionalEncoding",
    "GaussianAttention",
    "MetricEncoderWithGaussianAttention",
    "create_metric_encoder",
    # Log Encoder
    "LogEncoder",
    "SimplifiedLogEncoder",
    # Trace Encoder
    "TraceEncoder",
    "SimpleTraceEncoder",
]
