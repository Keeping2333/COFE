"""
Fusion module for multimodal embedding combination.

Provides:
- MultiModalProjector: Combines embeddings via concat/attention/gated fusion
- LayerAwareProjector: Projects to 4-layer fault space
"""
from .projector import LayerAwareProjector, MultiModalProjector

__all__ = [
    "MultiModalProjector",
    "LayerAwareProjector",
]
