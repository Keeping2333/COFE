"""
Loss functions for contrastive learning.

Provides:
- LayerContrastiveLoss: Same-layer positives, different-layer negatives
- HierarchicalContrastiveLoss: Layer similarity-aware contrastive loss
- TripletMarginLoss: Standard triplet loss with margin
- CombinedContrastiveLoss: Multi-objective loss combination
- FaultTypeContrastiveLoss: 12-class fault-type contrastive with hierarchical layer weighting
"""
from .contrastive import (
    CombinedContrastiveLoss,
    FaultTypeContrastiveLoss,
    HierarchicalContrastiveLoss,
    LayerContrastiveLoss,
    TripletMarginLoss,
)

__all__ = [
    "LayerContrastiveLoss",
    "HierarchicalContrastiveLoss",
    "TripletMarginLoss",
    "CombinedContrastiveLoss",
    "FaultTypeContrastiveLoss",
]
