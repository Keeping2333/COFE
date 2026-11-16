"""
Trace Encoder: Graph Attention Network (GAT) for distributed trace encoding.

Traces are naturally graphs where:
- Nodes = spans (individual operations/calls)
- Edges = parent-child relationships (CHILD_OF references)

The GAT architecture allows the model to:
- Aggregate information from related spans
- Learn which span relationships are important for RCA
- Handle variable-size trace graphs
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import EncoderOutput, LayerAwareEncoder


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention layer.

    Implements attention mechanism over graph neighbors:
    - Each node attends to its neighbors
    - Attention weights are learned based on node features
    - Multi-head attention for richer representations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        """
        Initialize GAT layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            concat: If True, concatenate head outputs; else average
            negative_slope: LeakyReLU negative slope
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention parameters (source and target)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_tgt = nn.Parameter(torch.empty(num_heads, out_features, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_tgt)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

        # Output dimension
        if concat:
            self.output_dim = out_features * num_heads
        else:
            self.output_dim = out_features

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of GAT layer.

        Args:
            x: Node features [batch, num_nodes, in_features]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, num_nodes, output_dim]
            - attention_weights: [batch, num_heads, num_nodes, num_nodes] if return_attention
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation: [batch, num_heads, num_nodes, out_features]
        h = torch.einsum('bni,hio->bhno', x, self.W)

        # Compute attention scores
        # Source attention: [batch, num_heads, num_nodes, 1]
        attn_src = torch.einsum('bhno,hol->bhnl', h, self.a_src)
        # Target attention: [batch, num_heads, num_nodes, 1]
        attn_tgt = torch.einsum('bhno,hol->bhnl', h, self.a_tgt)

        # Combine: attn_src[i] + attn_tgt[j] for edge (i, j)
        # [batch, num_heads, num_nodes, num_nodes]
        attn = attn_src + attn_tgt.transpose(-2, -1)
        attn = self.leaky_relu(attn)

        # Mask non-edges with large negative value
        # Expand adjacency for heads: [batch, 1, num_nodes, num_nodes]
        adj_expanded = adj.unsqueeze(1)
        attn = attn.masked_fill(adj_expanded == 0, -1e9)

        # Softmax over neighbors
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate neighbor features
        # [batch, num_heads, num_nodes, out_features]
        out = torch.einsum('bhmn,bhno->bhmo', attn, h)

        if self.concat:
            # Concatenate heads: [batch, num_nodes, num_heads * out_features]
            out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        else:
            # Average heads: [batch, num_nodes, out_features]
            out = out.mean(dim=1)

        if return_attention:
            return out, attn
        return out, None


class TraceEncoder(LayerAwareEncoder[Dict[str, Tensor]]):
    """
    Trace encoder using Graph Attention Network.

    Encodes distributed traces where spans are nodes and parent-child
    relationships are edges. Uses multi-layer GAT to aggregate
    information across the trace graph.

    Input format:
        {
            "span_features": [batch, max_spans, span_dim] - node features
            "trace_adjacency": [batch, max_spans, max_spans] - adjacency matrix
            "span_mask": [batch, max_spans] - which spans are valid (optional)
        }

    Output:
        EncoderOutput with:
        - embeddings: [batch, output_dim] - trace-level embedding
        - layer_assignments: [batch, 4] - fault layer probabilities
    """

    def __init__(
        self,
        span_dim: int = 5,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        enable_rich_features: bool = False,
        num_service_buckets: int = 64,
        num_operation_buckets: int = 1024,
        service_embed_dim: int = 16,
        operation_embed_dim: int = 32,
        num_status_classes: int = 4,
        status_embed_dim: int = 4,
        num_error_type_buckets: int = 32,
        error_type_embed_dim: int = 8,
        num_otel_status_classes: int = 3,
        otel_status_embed_dim: int = 4,
    ) -> None:
        """
        Initialize trace encoder.

        Args:
            span_dim: Dimension of input span features
            hidden_dim: Hidden dimension for GAT layers
            output_dim: Output embedding dimension
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads per GAT layer
            dropout: Dropout probability
            enable_rich_features: When True, augment span_features with
                learned embeddings of service_name, operation_name,
                status_code_class, and two boolean flags (is_root, is_leaf).
                Default False — encoder behaviour is byte-identical to
                pre-modification. Requires the dataset to also be constructed
                with enable_rich_span_features=True so the categorical IDs
                reach the encoder via the inputs dict.
            num_service_buckets: hash bucket count for service_name embedding
            num_operation_buckets: hash bucket count for operation_name embedding
            service_embed_dim: dim of the service_name embedding
            operation_embed_dim: dim of the operation_name embedding
            num_status_classes: number of HTTP status classes (4: none/2xx/4xx/5xx)
            status_embed_dim: dim of the status_class embedding
        """
        super().__init__(
            input_dim=span_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gat_layers,
            dropout=dropout,
        )

        self.span_dim = span_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads

        # Rich-feature plumbing: embedding tables are only allocated when
        # enabled, so the state_dict is byte-identical to pre-modification
        # in the default off path.
        self._enable_rich_features = enable_rich_features
        if enable_rich_features:
            self._service_embed = nn.Embedding(num_service_buckets, service_embed_dim)
            self._operation_embed = nn.Embedding(num_operation_buckets, operation_embed_dim)
            self._status_embed = nn.Embedding(num_status_classes, status_embed_dim)
            # Round2-only OTel error tags.
            self._error_type_embed = nn.Embedding(num_error_type_buckets, error_type_embed_dim)
            self._otel_status_embed = nn.Embedding(num_otel_status_classes, otel_status_embed_dim)
            # Raw scalars concatenated alongside embeddings:
            # is_root, is_leaf, is_on_critical_path, error_flag = 4
            enriched_input_dim = (
                span_dim
                + service_embed_dim + operation_embed_dim + status_embed_dim
                + error_type_embed_dim + otel_status_embed_dim
                + 4
            )
        else:
            enriched_input_dim = span_dim
        self._enriched_input_dim = enriched_input_dim

        # Input projection (input dim grows when rich features are enabled).
        self._input_proj = nn.Linear(enriched_input_dim, hidden_dim)

        # GAT layers
        self._gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim // num_heads

            # Last layer doesn't concatenate
            concat = (i < num_gat_layers - 1)
            if not concat:
                out_dim = hidden_dim

            self._gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )

        # Layer normalization
        self._layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])

        # Graph-level attention for pooling
        self._graph_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @property
    def modality(self) -> str:
        return "trace"

    def encode(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Encode trace graph using GAT layers.

        Args:
            inputs: Dict with span_features and trace_adjacency. When the
                encoder was constructed with enable_rich_features=True the
                following additional keys are consumed:
                  - span_service_ids:    [batch, num_nodes] long
                  - span_operation_ids:  [batch, num_nodes] long
                  - span_status_classes: [batch, num_nodes] long
                  - span_is_root:        [batch, num_nodes] long (0/1)
                  - span_is_leaf:        [batch, num_nodes] long (0/1)
                If the encoder is rich-enabled but the inputs dict does not
                carry these keys, we fall back to plain span_features (this
                lets us run forward-pass sanity tests without the dataset).

        Returns:
            Node-level hidden states [batch, num_nodes, hidden_dim]
        """
        span_features = inputs["span_features"]
        adjacency = inputs["trace_adjacency"]

        # Add self-loops to adjacency
        batch_size, num_nodes, _ = span_features.shape
        eye = torch.eye(num_nodes, device=adjacency.device).unsqueeze(0)
        adjacency = adjacency + eye

        # Concatenate rich categorical embeddings when enabled.
        if self._enable_rich_features:
            service_ids = inputs.get("span_service_ids")
            operation_ids = inputs.get("span_operation_ids")
            status_classes = inputs.get("span_status_classes")
            is_root = inputs.get("span_is_root")
            is_leaf = inputs.get("span_is_leaf")
            is_on_critical_path = inputs.get("span_is_on_critical_path")
            error_flag = inputs.get("span_error_flag")
            error_type_ids = inputs.get("span_error_type_ids")
            otel_status_classes = inputs.get("span_otel_status_classes")
            if service_ids is not None and operation_ids is not None and status_classes is not None:
                service_emb = self._service_embed(service_ids)
                operation_emb = self._operation_embed(operation_ids)
                status_emb = self._status_embed(status_classes)
                _zeros = lambda: torch.zeros(
                    batch_size, num_nodes,
                    device=span_features.device, dtype=torch.long,
                )
                is_root_f = (is_root if is_root is not None else _zeros()).float().unsqueeze(-1)
                is_leaf_f = (is_leaf if is_leaf is not None else _zeros()).float().unsqueeze(-1)
                is_cp_f = (is_on_critical_path if is_on_critical_path is not None
                           else _zeros()).float().unsqueeze(-1)
                error_flag_f = (error_flag if error_flag is not None
                                else _zeros()).float().unsqueeze(-1)
                # Categorical embeddings for round2 OTel error tags. Default
                # to bucket 0 / class 0 (UNSET) when the dataset doesn't ship them.
                error_type_emb = self._error_type_embed(
                    error_type_ids if error_type_ids is not None else _zeros()
                )
                otel_status_emb = self._otel_status_embed(
                    otel_status_classes if otel_status_classes is not None else _zeros()
                )
                span_features = torch.cat(
                    [
                        span_features, service_emb, operation_emb, status_emb,
                        error_type_emb, otel_status_emb,
                        is_root_f, is_leaf_f, is_cp_f, error_flag_f,
                    ],
                    dim=-1,
                )

        # Input projection
        h = self._input_proj(span_features)

        # GAT layers with residual connections
        for i, (gat, ln) in enumerate(zip(self._gat_layers, self._layer_norms)):
            h_prev = h
            h, _ = gat(h, adjacency)
            h = F.elu(h)

            # Residual connection (if dimensions match)
            if h.shape == h_prev.shape:
                h = h + h_prev

            h = ln(h)

        return h

    def pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pool node representations to graph-level using attention.

        Args:
            hidden_states: [batch, num_nodes, hidden_dim]
            attention_mask: Optional [batch, num_nodes] - valid node mask

        Returns:
            Graph-level representation [batch, hidden_dim]
        """
        # Compute attention scores
        attn_scores = self._graph_attention(hidden_states).squeeze(-1)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        # Normalize
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        pooled = torch.einsum('bn,bnd->bd', attn_weights, hidden_states)

        return pooled

    def forward(
        self,
        inputs: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
    ) -> EncoderOutput:
        """
        Full forward pass.

        Args:
            inputs: Dict with span_features and trace_adjacency
            attention_mask: Optional span mask

        Returns:
            EncoderOutput
        """
        # Use span_mask if provided in inputs
        if attention_mask is None:
            attention_mask = inputs.get("span_mask")

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


class SimpleTraceEncoder(LayerAwareEncoder[Dict[str, Tensor]]):
    """
    Simplified trace encoder without GAT.

    Uses attention over span features directly.
    Suitable for scenarios where graph structure is less important
    or for faster training.
    """

    def __init__(
        self,
        span_dim: int = 5,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=span_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self._input_proj = nn.Linear(span_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self._attention = nn.Linear(hidden_dim, 1)

    @property
    def modality(self) -> str:
        return "trace"

    def encode(self, inputs: Dict[str, Tensor]) -> Tensor:
        span_features = inputs["span_features"]
        span_mask = inputs.get("span_mask")

        h = self._input_proj(span_features)

        if span_mask is not None:
            src_key_padding_mask = span_mask == 0
        else:
            src_key_padding_mask = None

        h = self._transformer(h, src_key_padding_mask=src_key_padding_mask)
        return h

    def pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn_scores = self._attention(hidden_states).squeeze(-1)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.einsum('bn,bnd->bd', attn_weights, hidden_states)

        return pooled
