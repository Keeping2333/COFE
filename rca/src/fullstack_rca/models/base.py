"""
Abstract base classes for all encoders in the Full-stack RCA system.

This module defines the interface contract that all modality-specific
encoders (Log, Metric, Trace) must implement.

Key Design Principles:
1. All encoders produce embeddings in a common semantic space
2. LayerAwareEncoder variant adds soft assignment to fault layers (L1-L4)
3. Consistent output format via EncoderOutput dataclass
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor


class FaultLayer(Enum):
    """The four fault layers in the microservice stack."""
    L1_INFRASTRUCTURE = "L1"
    L2_NETWORK = "L2"
    L3_PLATFORM = "L3"
    L4_APPLICATION = "L4"

    @classmethod
    def from_string(cls, s: str) -> "FaultLayer":
        """Convert string like 'L1' to FaultLayer enum."""
        mapping = {
            "L1": cls.L1_INFRASTRUCTURE,
            "L2": cls.L2_NETWORK,
            "L3": cls.L3_PLATFORM,
            "L4": cls.L4_APPLICATION,
        }
        return mapping.get(s.upper(), cls.L4_APPLICATION)

    def to_index(self) -> int:
        """Convert to 0-indexed integer."""
        return int(self.value[1]) - 1


@dataclass
class EncoderOutput:
    """
    Standard output format for all encoders.

    This dataclass ensures consistent output across different modality
    encoders, enabling seamless fusion in the MultiModalProjector.

    Attributes:
        embeddings: The encoded representations.
            Shape depends on encoder:
            - [batch_size, output_dim] for pooled encoders
            - [batch_size, seq_len, output_dim] for sequence encoders
        pooled: Pooled representation [batch_size, output_dim].
            Always a fixed-size vector per sample.
        hidden_states: Optional intermediate hidden states.
            Useful for attention visualization and debugging.
        attention_weights: Optional attention weights for interpretability.
        layer_assignments: Optional soft assignment to fault layers (L1-L4).
            Shape: [batch_size, 4] with softmax probabilities.
    """
    embeddings: Tensor
    pooled: Tensor
    hidden_states: Optional[Tensor] = None
    attention_weights: Optional[Tensor] = None
    layer_assignments: Optional[Tensor] = None


# Type variable for modality-specific input
T = TypeVar('T')


class BaseEncoder(nn.Module, ABC, Generic[T]):
    """
    Abstract base class for all modality-specific encoders.

    All encoders must produce embeddings in a common semantic space
    to enable cross-modal contrastive learning and fusion.

    Type Parameters:
        T: The input type specific to each modality
           - LogEncoder: Dict with 'tokens', 'mask'
           - MetricEncoder: Dict with 'values', 'relative_time_positions'
           - TraceEncoder: Dict with 'span_features', 'adjacency'

    Subclasses must implement:
        - encode(): Modality-specific encoding logic
        - pool(): Sequence to vector pooling
        - modality: Property returning modality name
    """

    # Number of fault layers
    NUM_LAYERS = 4

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the base encoder.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings (shared across modalities)
            num_layers: Number of encoder layers
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Projection head to common embedding space
        self._projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Layer normalization for output
        self._output_norm = nn.LayerNorm(output_dim)

    @abstractmethod
    def encode(self, inputs: T) -> Tensor:
        """
        Encode modality-specific inputs to hidden representations.

        This method must be implemented by each encoder subclass
        to handle the specific preprocessing and encoding for
        that modality (logs, metrics, traces).

        Args:
            inputs: Modality-specific input data

        Returns:
            Hidden representations [batch_size, seq_len, hidden_dim]
        """
        pass

    @abstractmethod
    def pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pool sequence of hidden states to single vector.

        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Optional mask for valid positions [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_dim]
        """
        pass

    def project(self, pooled: Tensor) -> Tensor:
        """
        Project pooled representation to shared embedding space.

        Args:
            pooled: Pooled hidden state [batch_size, hidden_dim]

        Returns:
            Projected embedding [batch_size, output_dim]
        """
        projected = self._projection_head(pooled)
        return self._output_norm(projected)

    def forward(
        self,
        inputs: T,
        attention_mask: Optional[Tensor] = None,
    ) -> EncoderOutput:
        """
        Full forward pass: encode -> pool -> project.

        Args:
            inputs: Modality-specific input data
            attention_mask: Optional attention mask

        Returns:
            EncoderOutput with embeddings and pooled representation
        """
        hidden_states = self.encode(inputs)
        pooled = self.pool(hidden_states, attention_mask)
        embeddings = self.project(pooled)

        return EncoderOutput(
            embeddings=embeddings,
            pooled=pooled,
            hidden_states=hidden_states,
        )

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the modality name ('log', 'metric', 'trace')."""
        pass

    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim


class LayerAwareEncoder(BaseEncoder[T], ABC):
    """
    Extension of BaseEncoder with layer-aware (L1-L4) encoding.

    This encoder variant produces not only embeddings but also
    soft assignments to the 4 fault layers, enabling hierarchical
    contrastive learning and layer-aware RCA.

    The layer classifier is trained jointly with the encoder
    using the supervised layer labels from fault injection.
    """

    LAYER_NAMES = ['infrastructure', 'network', 'platform', 'application']

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize layer-aware encoder.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of encoder layers
            dropout: Dropout probability
        """
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout)

        # Layer classification head
        # Maps pooled representation to 4 layer probabilities
        self._layer_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.NUM_LAYERS),
        )

    def classify_layer(self, pooled: Tensor) -> Tensor:
        """
        Produce soft assignment probabilities over 4 fault layers.

        Args:
            pooled: Pooled representation [batch_size, hidden_dim]

        Returns:
            Layer probabilities [batch_size, 4] (softmax over L1-L4)
        """
        logits = self._layer_classifier(pooled)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        inputs: T,
        attention_mask: Optional[Tensor] = None,
    ) -> EncoderOutput:
        """
        Forward pass with layer assignment.

        Args:
            inputs: Modality-specific input data
            attention_mask: Optional attention mask

        Returns:
            EncoderOutput with embeddings, pooled, and layer_assignments
        """
        hidden_states = self.encode(inputs)
        pooled = self.pool(hidden_states, attention_mask)
        embeddings = self.project(pooled)
        layer_probs = self.classify_layer(pooled)

        return EncoderOutput(
            embeddings=embeddings,
            pooled=pooled,
            hidden_states=hidden_states,
            layer_assignments=layer_probs,
        )


class PoolingStrategy(Enum):
    """Available pooling strategies for sequence encoders."""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    ATTENTION = "attention"


def mean_pooling(
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Mean pooling over sequence dimension.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: Optional [batch_size, seq_len]

    Returns:
        Pooled tensor [batch_size, hidden_dim]
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    # Expand mask for broadcasting
    mask = attention_mask.unsqueeze(-1).float()

    # Masked mean
    sum_hidden = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)

    return sum_hidden / count


def max_pooling(
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Max pooling over sequence dimension.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: Optional [batch_size, seq_len]

    Returns:
        Pooled tensor [batch_size, hidden_dim]
    """
    if attention_mask is not None:
        # Set masked positions to large negative value
        mask = attention_mask.unsqueeze(-1).float()
        hidden_states = hidden_states.masked_fill(mask == 0, -1e9)

    return hidden_states.max(dim=1)[0]


def attention_pooling(
    hidden_states: Tensor,
    attention_weights: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Attention-weighted pooling over sequence dimension.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_weights: [batch_size, seq_len]
        attention_mask: Optional [batch_size, seq_len]

    Returns:
        Pooled tensor [batch_size, hidden_dim]
    """
    if attention_mask is not None:
        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)

    # Normalize attention weights
    attn = torch.softmax(attention_weights, dim=-1)

    # Weighted sum
    return torch.einsum('bs,bsd->bd', attn, hidden_states)
