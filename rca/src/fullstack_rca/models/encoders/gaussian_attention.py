"""
Gaussian Attention for time-aware metric encoding.

Implements the formula from FAMOS paper:
    G(m_i) = (1 / sqrt(2*pi) * sigma) * exp(-|t_metric_i - t_trace|^2 / (2 * sigma^2))

Where:
- t_metric_i is the timestamp of metric point i
- t_trace is the anchor trace timestamp
- sigma is a learnable parameter controlling temporal locality

This attention mechanism gives higher weights to metric points
closer in time to the anchor trace, implementing "soft" temporal
alignment between metrics and traces.

Key Design Decisions:
1. sigma is learnable to adapt temporal locality per task
2. Store log(sigma) for numerical stability (ensures sigma > 0)
3. Combine Gaussian weights with content attention for best results
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GaussianPositionalEncoding(nn.Module):
    """
    Gaussian-weighted positional encoding for metric time series.

    Given relative time positions (|t_metric - t_trace|), this module
    computes Gaussian weights that decay with temporal distance.

    The sigma parameter is learnable, allowing the model to adapt
    the temporal locality based on the data characteristics.

    Attributes:
        min_sigma: Minimum allowed sigma value (prevents collapse)
        max_sigma: Maximum allowed sigma value (prevents over-smoothing)
    """

    def __init__(
        self,
        initial_sigma: float = 10.0,
        learnable: bool = True,
        min_sigma: float = 0.1,
        max_sigma: float = 100.0,
    ) -> None:
        """
        Initialize Gaussian positional encoding.

        Args:
            initial_sigma: Initial value for sigma (in seconds).
                Higher values = broader attention window.
                Lower values = more focused on nearby timestamps.
            learnable: Whether sigma should be learnable via backprop.
            min_sigma: Minimum allowed sigma value (prevents division issues).
            max_sigma: Maximum allowed sigma value (prevents over-smoothing).
        """
        super().__init__()

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Store log(sigma) for numerical stability
        # This ensures sigma stays positive after exp()
        initial_log_sigma = math.log(initial_sigma)

        if learnable:
            self._log_sigma = nn.Parameter(torch.tensor(initial_log_sigma))
        else:
            self.register_buffer("_log_sigma", torch.tensor(initial_log_sigma))

        # Precompute normalization constant
        self._sqrt_2pi = math.sqrt(2.0 * math.pi)

    @property
    def sigma(self) -> Tensor:
        """Get the current sigma value, clamped to valid range."""
        sigma = torch.exp(self._log_sigma)
        return torch.clamp(sigma, self.min_sigma, self.max_sigma)

    def forward(self, relative_time_positions: Tensor) -> Tensor:
        """
        Compute Gaussian weights for relative time positions.

        Formula: G(t) = (1 / sqrt(2*pi) * sigma) * exp(-t^2 / (2 * sigma^2))

        Args:
            relative_time_positions: Tensor of shape [..., seq_len]
                containing |t_metric_i - t_trace| for each metric point.
                Values should be in seconds.

        Returns:
            Gaussian weights of shape [..., seq_len].
            Higher weights for positions closer to zero (anchor).
        """
        sigma = self.sigma

        # Compute Gaussian
        # G(t) = (1 / sqrt(2*pi) * sigma) * exp(-t^2 / (2 * sigma^2))
        normalization = 1.0 / (self._sqrt_2pi * sigma)
        exponent = -(relative_time_positions ** 2) / (2.0 * sigma ** 2)

        weights = normalization * torch.exp(exponent)

        return weights

    def normalize_weights(self, weights: Tensor, dim: int = -1) -> Tensor:
        """
        Normalize weights to sum to 1 along specified dimension.

        Useful when using Gaussian weights as attention scores.

        Args:
            weights: Gaussian weights
            dim: Dimension to normalize along

        Returns:
            Normalized weights summing to 1
        """
        return weights / (weights.sum(dim=dim, keepdim=True) + 1e-8)


class GaussianAttention(nn.Module):
    """
    Full Gaussian Attention module for metric encoding.

    Combines:
    1. Gaussian positional encoding (temporal locality)
    2. Content-based attention (learned from features)

    The final attention is a product of Gaussian weights and
    content attention, ensuring both temporal and semantic relevance.

    This implements the key insight: metrics close to the trace timestamp
    are more likely to be causally related to the observed behavior.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        initial_sigma: float = 10.0,
        dropout: float = 0.1,
        combine_mode: str = "multiply",
    ) -> None:
        """
        Initialize Gaussian Attention.

        Args:
            hidden_dim: Dimension of input features
            num_heads: Number of attention heads
            initial_sigma: Initial sigma for Gaussian encoding (seconds)
            dropout: Dropout probability
            combine_mode: How to combine Gaussian and content attention
                - "multiply": attention = softmax(content) * gaussian
                - "add": attention = softmax(content + log(gaussian))
                - "gaussian_only": attention = gaussian (no content)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.combine_mode = combine_mode

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Gaussian positional encoding (learnable sigma)
        self.gaussian_encoding = GaussianPositionalEncoding(
            initial_sigma=initial_sigma,
            learnable=True,
        )

        # Content attention components (Q, K, V projections)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        relative_time_positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply Gaussian-weighted attention.

        Args:
            query: Query tensor [batch, query_len, hidden_dim]
            key: Key tensor [batch, key_len, hidden_dim]
            value: Value tensor [batch, key_len, hidden_dim]
            relative_time_positions: Time deltas [batch, key_len]
                These are |t_metric - t_trace| values in seconds.
            attention_mask: Optional mask [batch, key_len]
                1 for valid positions, 0 for padding.

        Returns:
            Tuple of (output, attention_weights):
                - output: [batch, query_len, hidden_dim]
                - attention_weights: [batch, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project to multi-head format
        # [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
        Q = self.query_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute content attention scores
        # [batch, num_heads, query_len, key_len]
        content_attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Compute Gaussian weights based on time positions
        # Shape: [batch, key_len]
        gaussian_weights = self.gaussian_encoding(relative_time_positions)
        gaussian_weights = self.gaussian_encoding.normalize_weights(gaussian_weights)

        # Expand for multi-head: [batch, 1, 1, key_len]
        gaussian_weights = gaussian_weights.unsqueeze(1).unsqueeze(2)

        # Combine Gaussian and content attention based on mode
        if self.combine_mode == "multiply":
            # Multiplicative: attention = softmax(content) * gaussian
            # This ensures high attention only when BOTH content and time align
            content_attn = torch.softmax(content_attn, dim=-1)
            attention = content_attn * gaussian_weights

        elif self.combine_mode == "add":
            # Additive: add log(gaussian) to logits before softmax
            # This biases attention toward temporally close positions
            log_gaussian = torch.log(gaussian_weights + 1e-8)
            attention = torch.softmax(content_attn + log_gaussian, dim=-1)

        elif self.combine_mode == "gaussian_only":
            # Only Gaussian weights, no content attention
            # Pure temporal proximity weighting
            attention = gaussian_weights.expand(-1, self.num_heads, query_len, -1)

        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, key_len] -> [batch, 1, 1, key_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention = attention.masked_fill(attention_mask == 0, 0.0)

        # Re-normalize attention weights
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        attention = self.dropout(attention)

        # Apply attention to values
        # [batch, num_heads, query_len, head_dim]
        output = torch.matmul(attention, V)

        # Reshape back to [batch, query_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        output = self.output_proj(output)

        return output, attention


class MetricEncoderWithGaussianAttention(nn.Module):
    """
    Complete metric encoder using 1D-CNN + Gaussian Attention.

    Architecture:
    1. 1D-CNN for local feature extraction from metric time series
    2. Gaussian Attention for temporal alignment with anchor trace
    3. Pooling to produce final metric embedding

    This encoder addresses the key challenge of aligning periodic
    metric data with event-driven traces by using learnable
    Gaussian attention to weight metrics by temporal proximity.
    """

    def __init__(
        self,
        num_metrics: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_cnn_layers: int = 3,
        kernel_size: int = 3,
        num_attention_heads: int = 4,
        initial_sigma: float = 10.0,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize metric encoder.

        Args:
            num_metrics: Number of metric types (input channels).
                E.g., 44 for the train-ticket dataset.
            hidden_dim: Hidden dimension for CNN and attention.
            output_dim: Output embedding dimension.
            num_cnn_layers: Number of 1D CNN layers.
            kernel_size: CNN kernel size (should be odd for same padding).
            num_attention_heads: Number of attention heads.
            initial_sigma: Initial sigma for Gaussian attention (seconds).
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_metrics = num_metrics
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Build 1D-CNN for local feature extraction
        cnn_layers = []
        in_channels = num_metrics

        for i in range(num_cnn_layers):
            out_channels = hidden_dim if i == num_cnn_layers - 1 else hidden_dim // 2
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Gaussian Attention for temporal alignment
        self.gaussian_attention = GaussianAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            initial_sigma=initial_sigma,
            dropout=dropout,
            combine_mode="multiply",  # Best empirical results
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        metric_values: Tensor,
        relative_time_positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode metric time series with Gaussian attention.

        Args:
            metric_values: Metric time series [batch, num_metrics, seq_len]
            relative_time_positions: [batch, seq_len] containing
                |t_metric - t_trace| for each time point (in seconds).
            attention_mask: Optional [batch, seq_len] mask.
                1 for valid positions, 0 for padding.

        Returns:
            Tuple of (embeddings, attention_weights):
                - embeddings: [batch, output_dim] final metric embedding
                - attention_weights: [batch, num_heads, 1, seq_len] for visualization
        """
        batch_size, num_metrics, seq_len = metric_values.shape

        # 1. 1D-CNN feature extraction
        # Input: [batch, num_metrics, seq_len]
        # Output: [batch, hidden_dim, seq_len]
        cnn_features = self.cnn(metric_values)

        # Transpose for attention: [batch, seq_len, hidden_dim]
        cnn_features = cnn_features.transpose(1, 2)

        # 2. Create query as mean of features (for pooling attention)
        # We use a single query that attends to all time steps
        # [batch, 1, hidden_dim]
        query = cnn_features.mean(dim=1, keepdim=True)

        # 3. Apply Gaussian Attention
        # Query attends to all time steps with Gaussian temporal weighting
        attended, attention_weights = self.gaussian_attention(
            query=query,
            key=cnn_features,
            value=cnn_features,
            relative_time_positions=relative_time_positions,
            attention_mask=attention_mask,
        )

        # Remove query dimension: [batch, hidden_dim]
        attended = attended.squeeze(1)

        # 4. Layer norm and output projection
        attended = self.layer_norm(attended)
        embeddings = self.output_proj(attended)

        return embeddings, attention_weights

    @property
    def sigma(self) -> Tensor:
        """Get current learned sigma value for inspection."""
        return self.gaussian_attention.gaussian_encoding.sigma


# Convenience function for creating the encoder
def create_metric_encoder(
    num_metrics: int = 44,
    hidden_dim: int = 256,
    output_dim: int = 128,
    initial_sigma: float = 10.0,
) -> MetricEncoderWithGaussianAttention:
    """
    Factory function to create a metric encoder with default settings.

    Args:
        num_metrics: Number of metric types (default 44 for train-ticket)
        hidden_dim: Hidden dimension
        output_dim: Output embedding dimension
        initial_sigma: Initial Gaussian sigma in seconds

    Returns:
        Configured MetricEncoderWithGaussianAttention instance
    """
    return MetricEncoderWithGaussianAttention(
        num_metrics=num_metrics,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        initial_sigma=initial_sigma,
    )
