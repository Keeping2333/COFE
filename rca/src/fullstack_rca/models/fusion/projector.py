"""
MultiModal Projector: Fuses embeddings from all modalities into 4-Layer Vector Space.

This module takes encoded representations from:
- Log encoder (semantic patterns)
- Metric encoder (temporal signals with Gaussian attention)
- Trace encoder (structural/topological information)

And projects them into a unified 4-layer vector space corresponding to:
- L1: Infrastructure (hardware, host-level)
- L2: Network (communication, service-to-service)
- L3: Platform (orchestration, Kubernetes)
- L4: Application (business logic, code-level)

The projection enables:
1. Cross-modal contrastive learning
2. Layer-aware anomaly classification
3. Structured reasoning for RCA
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import EncoderOutput


class MultiModalProjector(nn.Module):
    """
    Projects and fuses multimodal embeddings into unified 4-layer space.

    Architecture options:
    1. Concatenation + MLP
    2. Cross-modal attention
    3. Gated fusion

    The output is a fused representation that captures information
    from all modalities and can be used for:
    - Layer classification (L1-L4)
    - Contrastive learning
    - LLM-based reasoning input
    """

    def __init__(
        self,
        log_dim: int = 128,
        metric_dim: int = 128,
        trace_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 4,  # L1-L4
        fusion_method: str = "concat",
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize multimodal projector.

        Args:
            log_dim: Dimension of log encoder output
            metric_dim: Dimension of metric encoder output
            trace_dim: Dimension of trace encoder output
            hidden_dim: Hidden dimension for fusion layers
            output_dim: Final output dimension
            num_layers: Number of fault layers (default 4)
            fusion_method: How to fuse modalities
                - "concat": Simple concatenation + MLP
                - "attention": Cross-modal attention fusion
                - "gated": Gated fusion mechanism
            dropout: Dropout probability
        """
        super().__init__()

        self.log_dim = log_dim
        self.metric_dim = metric_dim
        self.trace_dim = trace_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.fusion_method = fusion_method

        # Total input dimension
        self.total_dim = log_dim + metric_dim + trace_dim

        # Build fusion network based on method
        if fusion_method == "concat":
            self._build_concat_fusion(dropout)
        elif fusion_method == "attention":
            self._build_attention_fusion(dropout)
        elif fusion_method == "gated":
            self._build_gated_fusion(dropout)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Layer-specific projection heads
        # Each head projects the fused representation to a layer-specific space
        self._layer_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim // num_layers),
            )
            for _ in range(num_layers)
        ])

        # Layer classification head
        self._layer_classifier = nn.Linear(hidden_dim, num_layers)

        # Final output projection
        self._output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _build_concat_fusion(self, dropout: float) -> None:
        """Build concatenation-based fusion."""
        self._fusion = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hidden_dim),
        )

    def _build_attention_fusion(self, dropout: float) -> None:
        """Build cross-modal attention fusion."""
        # Project each modality to common dimension
        self._log_proj = nn.Linear(self.log_dim, self.hidden_dim)
        self._metric_proj = nn.Linear(self.metric_dim, self.hidden_dim)
        self._trace_proj = nn.Linear(self.trace_dim, self.hidden_dim)

        # Cross-modal attention
        self._cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self._fusion_norm = nn.LayerNorm(self.hidden_dim)

    def _build_gated_fusion(self, dropout: float) -> None:
        """Build gated fusion mechanism."""
        # Individual modality projections
        self._log_proj = nn.Linear(self.log_dim, self.hidden_dim)
        self._metric_proj = nn.Linear(self.metric_dim, self.hidden_dim)
        self._trace_proj = nn.Linear(self.trace_dim, self.hidden_dim)

        # Gating networks
        self._gate_log = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, 1),
            nn.Sigmoid(),
        )
        self._gate_metric = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, 1),
            nn.Sigmoid(),
        )
        self._gate_trace = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, 1),
            nn.Sigmoid(),
        )

        self._fusion_norm = nn.LayerNorm(self.hidden_dim)
        self._dropout = nn.Dropout(dropout)

    def fuse_concat(
        self,
        log_emb: Tensor,
        metric_emb: Tensor,
        trace_emb: Tensor,
    ) -> Tensor:
        """Fuse by concatenation + MLP."""
        # Concatenate all modalities
        concat = torch.cat([log_emb, metric_emb, trace_emb], dim=-1)
        return self._fusion(concat)

    def fuse_attention(
        self,
        log_emb: Tensor,
        metric_emb: Tensor,
        trace_emb: Tensor,
    ) -> Tensor:
        """Fuse using cross-modal attention."""
        batch_size = log_emb.shape[0]

        # Project to common dimension
        log_h = self._log_proj(log_emb)
        metric_h = self._metric_proj(metric_emb)
        trace_h = self._trace_proj(trace_emb)

        # Stack as sequence: [batch, 3, hidden_dim]
        modal_seq = torch.stack([log_h, metric_h, trace_h], dim=1)

        # Self-attention across modalities
        attn_out, _ = self._cross_attention(
            modal_seq, modal_seq, modal_seq
        )

        # Mean pool across modalities
        fused = attn_out.mean(dim=1)

        return self._fusion_norm(fused)

    def fuse_gated(
        self,
        log_emb: Tensor,
        metric_emb: Tensor,
        trace_emb: Tensor,
    ) -> Tensor:
        """Fuse using gated mechanism."""
        # Project to common dimension
        log_h = self._log_proj(log_emb)
        metric_h = self._metric_proj(metric_emb)
        trace_h = self._trace_proj(trace_emb)

        # Compute gates based on all modalities
        all_h = torch.cat([log_h, metric_h, trace_h], dim=-1)

        gate_log = self._gate_log(all_h)
        gate_metric = self._gate_metric(all_h)
        gate_trace = self._gate_trace(all_h)

        # Gated combination
        fused = gate_log * log_h + gate_metric * metric_h + gate_trace * trace_h

        fused = self._fusion_norm(fused)
        fused = self._dropout(fused)

        return fused

    def forward(
        self,
        log_embedding: Tensor,
        metric_embedding: Tensor,
        trace_embedding: Tensor,
        return_layer_embeddings: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Fuse multimodal embeddings.

        Args:
            log_embedding: [batch, log_dim] from LogEncoder
            metric_embedding: [batch, metric_dim] from MetricEncoder
            trace_embedding: [batch, trace_dim] from TraceEncoder
            return_layer_embeddings: Whether to return per-layer embeddings

        Returns:
            Dict with:
            - "fused": [batch, output_dim] - unified embedding
            - "layer_logits": [batch, 4] - layer classification logits
            - "layer_probs": [batch, 4] - layer probabilities (softmax)
            - "layer_embeddings": Optional [batch, 4, output_dim//4] - per-layer embeddings
        """
        # Fuse modalities
        if self.fusion_method == "concat":
            fused_hidden = self.fuse_concat(log_embedding, metric_embedding, trace_embedding)
        elif self.fusion_method == "attention":
            fused_hidden = self.fuse_attention(log_embedding, metric_embedding, trace_embedding)
        elif self.fusion_method == "gated":
            fused_hidden = self.fuse_gated(log_embedding, metric_embedding, trace_embedding)

        # Layer classification
        layer_logits = self._layer_classifier(fused_hidden)
        layer_probs = F.softmax(layer_logits, dim=-1)

        # Final output embedding
        fused_output = self._output_proj(fused_hidden)

        result = {
            "fused": fused_output,
            "fused_hidden": fused_hidden,
            "layer_logits": layer_logits,
            "layer_probs": layer_probs,
        }

        # Optionally compute per-layer embeddings
        if return_layer_embeddings:
            layer_embeddings = []
            for i, proj in enumerate(self._layer_projectors):
                layer_emb = proj(fused_hidden)
                layer_embeddings.append(layer_emb)

            # Stack: [batch, num_layers, output_dim // num_layers]
            result["layer_embeddings"] = torch.stack(layer_embeddings, dim=1)

        return result

    def forward_from_encoder_outputs(
        self,
        log_output: EncoderOutput,
        metric_output: EncoderOutput,
        trace_output: EncoderOutput,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass using EncoderOutput objects.

        Convenience method that extracts embeddings from encoder outputs.

        Args:
            log_output: Output from LogEncoder
            metric_output: Output from MetricEncoder
            trace_output: Output from TraceEncoder
            **kwargs: Additional arguments for forward()

        Returns:
            Fusion result dict
        """
        return self.forward(
            log_embedding=log_output.embeddings,
            metric_embedding=metric_output.embeddings,
            trace_embedding=trace_output.embeddings,
            **kwargs,
        )


class LayerAwareProjector(nn.Module):
    """
    Projects fused embeddings into 4-layer vector space.

    Each layer gets a dedicated subspace, and the final representation
    is structured as [L1_emb, L2_emb, L3_emb, L4_emb].

    This enables:
    - Layer-specific contrastive learning
    - Interpretable layer contributions
    - Structured input for LLM reasoning
    """

    LAYER_NAMES = ['infrastructure', 'network', 'platform', 'application']

    def __init__(
        self,
        input_dim: int,
        layer_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize layer-aware projector.

        Args:
            input_dim: Input dimension from fusion
            layer_dim: Dimension per layer embedding
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.output_dim = layer_dim * 4

        # Per-layer projection heads
        self._layer_heads = nn.ModuleDict({
            f"L{i+1}": nn.Sequential(
                nn.Linear(input_dim, layer_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_dim * 2, layer_dim),
                nn.LayerNorm(layer_dim),
            )
            for i in range(4)
        })

        # Layer attention (how much each layer contributes to the anomaly)
        self._layer_attention = nn.Linear(input_dim, 4)

    def forward(self, fused_hidden: Tensor) -> Dict[str, Tensor]:
        """
        Project to 4-layer vector space.

        Args:
            fused_hidden: [batch, input_dim] from MultiModalProjector

        Returns:
            Dict with:
            - "layer_vectors": [batch, 4, layer_dim] - per-layer embeddings
            - "full_embedding": [batch, 4 * layer_dim] - concatenated
            - "layer_attention": [batch, 4] - layer importance weights
        """
        batch_size = fused_hidden.shape[0]

        # Compute per-layer embeddings
        layer_vectors = []
        for i in range(4):
            layer_emb = self._layer_heads[f"L{i+1}"](fused_hidden)
            layer_vectors.append(layer_emb)

        # Stack: [batch, 4, layer_dim]
        layer_vectors = torch.stack(layer_vectors, dim=1)

        # Concatenate for full embedding: [batch, 4 * layer_dim]
        full_embedding = layer_vectors.view(batch_size, -1)

        # Layer attention weights
        layer_attention = F.softmax(self._layer_attention(fused_hidden), dim=-1)

        return {
            "layer_vectors": layer_vectors,
            "full_embedding": full_embedding,
            "layer_attention": layer_attention,
        }
