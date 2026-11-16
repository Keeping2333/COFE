"""
Standalone layer classifier using pre-fusion modality embeddings.

Centroid analysis (analyze_centroids.py) showed:
- Metric encoder separates L1 from {L2, L3, L4}  (cos < -0.26 for all)
- Log encoder separates {L2, L3, L4} from each other  (cos < -0.48 for all)
- Trace encoder contributes no layer-discriminative signal  (cos > 0.93 for all)

This classifier concatenates log + metric embeddings (512D) and uses
a small MLP to predict the fault layer (L1-L4).
"""
import torch
import torch.nn as nn
from torch import Tensor


class LayerClassifier(nn.Module):
    """MLP classifier: log+metric embeddings → 4 fault layers."""

    def __init__(
        self,
        log_dim: int = 256,
        metric_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = log_dim + metric_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_layers),
        )

    def forward(self, log_emb: Tensor, metric_emb: Tensor) -> Tensor:
        """
        Args:
            log_emb: [batch, log_dim] from LogEncoder
            metric_emb: [batch, metric_dim] from MetricEncoder

        Returns:
            logits: [batch, num_layers]
        """
        x = torch.cat([log_emb, metric_emb], dim=-1)
        return self.mlp(x)
