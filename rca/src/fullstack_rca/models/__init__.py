"""
Models module for Full-stack RCA.

Provides:
- Base encoder classes and fault layer definitions
- Modality-specific encoders (Log, Metric, Trace)
- Multimodal fusion projectors
- Contrastive learning losses
"""
from .base import (
    BaseEncoder,
    EncoderOutput,
    FaultLayer,
    LayerAwareEncoder,
)
from .encoders import (
    GaussianAttention,
    GaussianPositionalEncoding,
    LogEncoder,
    MetricEncoderWithGaussianAttention,
    SimplifiedLogEncoder,
    SimpleTraceEncoder,
    TraceEncoder,
    create_metric_encoder,
)
from .fusion import (
    LayerAwareProjector,
    MultiModalProjector,
)
from .losses import (
    CombinedContrastiveLoss,
    HierarchicalContrastiveLoss,
    LayerContrastiveLoss,
    TripletMarginLoss,
)

__all__ = [
    # Base
    "FaultLayer",
    "EncoderOutput",
    "BaseEncoder",
    "LayerAwareEncoder",
    # Encoders
    "GaussianPositionalEncoding",
    "GaussianAttention",
    "MetricEncoderWithGaussianAttention",
    "create_metric_encoder",
    "LogEncoder",
    "SimplifiedLogEncoder",
    "TraceEncoder",
    "SimpleTraceEncoder",
    # Fusion
    "MultiModalProjector",
    "LayerAwareProjector",
    # Losses
    "LayerContrastiveLoss",
    "HierarchicalContrastiveLoss",
    "TripletMarginLoss",
    "CombinedContrastiveLoss",
]
