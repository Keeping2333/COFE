"""
Log Encoder: DistilBERT embedding.

DistilBERT produces semantic embeddings for log messages, letting the model
understand log semantics and handle unseen messages gracefully.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..base import BaseEncoder, EncoderOutput, LayerAwareEncoder, mean_pooling


class LogEncoder(LayerAwareEncoder[Dict[str, Tensor]]):
    """
    Log encoder using DistilBERT for embedding.

    Architecture:
    1. DistilBERT embedding (semantic understanding)
    2. Attention pooling over log sequence
    3. Projection to shared embedding space

    Input format:
        {
            "log_tokens": [batch, num_logs, max_len] - tokenized log messages
            "log_mask": [batch, num_logs, max_len] - attention mask
        }

    Output:
        EncoderOutput with:
        - embeddings: [batch, output_dim]
        - layer_assignments: [batch, 4]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 128,
        pretrained_model: str = "distilbert-base-uncased",
        freeze_bert: bool = False,
        unfreeze_last_n: int = 0,
        max_logs: int = 100,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize log encoder.

        Args:
            hidden_dim: Hidden dimension for projection layers
            output_dim: Output embedding dimension
            pretrained_model: HuggingFace model name for embeddings
            freeze_bert: Whether to freeze BERT weights
            unfreeze_last_n: If freeze_bert=True, unfreeze the last N transformer layers
                             (e.g., 2 = unfreeze layers 4,5 of DistilBERT's 6 layers).
                             Enables fine-tuning domain-specific features while keeping
                             early layers frozen.
            max_logs: Maximum number of logs per sample
            dropout: Dropout probability
        """
        # Get BERT hidden size
        bert_hidden = 768 if "base" in pretrained_model else 1024

        super().__init__(
            input_dim=bert_hidden,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.max_logs = max_logs
        self.pretrained_model_name = pretrained_model

        # Initialize BERT model
        if HAS_TRANSFORMERS:
            self._bert = AutoModel.from_pretrained(pretrained_model)

            if freeze_bert:
                for param in self._bert.parameters():
                    param.requires_grad = False

                # Selectively unfreeze last N transformer layers
                if unfreeze_last_n > 0 and hasattr(self._bert, 'transformer'):
                    layers = self._bert.transformer.layer
                    num_layers = len(layers)
                    for i in range(max(0, num_layers - unfreeze_last_n), num_layers):
                        for param in layers[i].parameters():
                            param.requires_grad = True
        else:
            # Fallback: simple embedding layer
            self._bert = None
            self._fallback_embedding = nn.Embedding(30522, bert_hidden)

        # Log-level attention for pooling multiple logs
        # Note: Input is already projected to hidden_dim by _input_proj in encode()
        self._log_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Projection from BERT hidden to our hidden dim
        self._input_proj = nn.Linear(bert_hidden, hidden_dim)

    @property
    def modality(self) -> str:
        return "log"

    def encode_tokens(
        self,
        log_tokens: Tensor,
        log_mask: Tensor,
    ) -> Tensor:
        """
        Encode tokenized log messages using BERT.

        Args:
            log_tokens: [batch, num_logs, seq_len]
            log_mask: [batch, num_logs, seq_len]

        Returns:
            Log embeddings [batch, num_logs, bert_hidden]
        """
        batch_size, num_logs, seq_len = log_tokens.shape

        # Flatten for BERT processing
        flat_tokens = log_tokens.view(-1, seq_len)  # [batch * num_logs, seq_len]
        flat_mask = log_mask.view(-1, seq_len)

        if self._bert is not None:
            # Use BERT
            with torch.set_grad_enabled(self.training):
                outputs = self._bert(
                    input_ids=flat_tokens,
                    attention_mask=flat_mask,
                    return_dict=True,
                )
                # Use [CLS] token representation
                log_embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            # Fallback: mean of embeddings
            embeddings = self._fallback_embedding(flat_tokens)
            log_embeddings = mean_pooling(embeddings, flat_mask)

        # Reshape back
        log_embeddings = log_embeddings.view(batch_size, num_logs, -1)

        return log_embeddings

    def encode(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Encode log inputs to hidden representations.

        Args:
            inputs: Dict with:
                - log_tokens: [batch, num_logs, seq_len]
                - log_mask: [batch, num_logs, seq_len]

        Returns:
            Hidden representations [batch, num_logs, hidden_dim]
        """
        log_tokens = inputs["log_tokens"]
        log_mask = inputs["log_mask"]

        # Get BERT embeddings
        log_embeddings = self.encode_tokens(log_tokens, log_mask)

        # Project to hidden dim
        hidden = self._input_proj(log_embeddings)

        return hidden

    def pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pool multiple log representations to single vector using attention.

        Args:
            hidden_states: [batch, num_logs, hidden_dim]
            attention_mask: Optional [batch, num_logs] - which logs are valid

        Returns:
            Pooled representation [batch, hidden_dim]
        """
        # Compute attention scores
        attn_scores = self._log_attention(hidden_states).squeeze(-1)  # [batch, num_logs]

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        # Normalize
        attn_weights = torch.softmax(attn_scores, dim=-1)

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
            inputs: Dict with log_tokens and log_mask
            attention_mask: Optional mask for which logs are valid

        Returns:
            EncoderOutput
        """
        hidden_states = self.encode(inputs)

        # Create log-level mask from token-level mask if not provided
        if attention_mask is None and "log_mask" in inputs:
            # A log is valid if any of its tokens are valid
            attention_mask = inputs["log_mask"].any(dim=-1).float()

        pooled = self.pool(hidden_states, attention_mask)
        embeddings = self.project(pooled)
        layer_probs = self.classify_layer(pooled)

        return EncoderOutput(
            embeddings=embeddings,
            pooled=pooled,
            hidden_states=hidden_states,
            layer_assignments=layer_probs,
        )


class SimplifiedLogEncoder(BaseEncoder[Dict[str, Tensor]]):
    """
    Simplified log encoder.

    Uses a trainable token embedding + Transformer encoder with attention
    pooling, instead of a pretrained DistilBERT backbone.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize simplified log encoder.

        Args:
            vocab_size: Vocabulary size for embedding layer
            embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.vocab_size = vocab_size

        # Token embedding
        self._embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding
        self._pos_encoding = nn.Embedding(512, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Log-level attention
        self._log_attention = nn.Linear(embedding_dim, 1)

    @property
    def modality(self) -> str:
        return "log"

    def encode(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Encode log inputs.

        Args:
            inputs: Dict with log_tokens [batch, num_logs, seq_len]

        Returns:
            Hidden states [batch, num_logs, hidden_dim]
        """
        log_tokens = inputs["log_tokens"]
        log_mask = inputs.get("log_mask")

        batch_size, num_logs, seq_len = log_tokens.shape

        # Flatten
        flat_tokens = log_tokens.view(-1, seq_len)

        # Embed tokens
        token_emb = self._embedding(flat_tokens)

        # Add positional encoding
        positions = torch.arange(seq_len, device=flat_tokens.device)
        pos_emb = self._pos_encoding(positions)
        embeddings = token_emb + pos_emb

        # Create attention mask for transformer
        if log_mask is not None:
            flat_mask = log_mask.view(-1, seq_len)
            src_key_padding_mask = flat_mask == 0
        else:
            src_key_padding_mask = None

        # Transform
        hidden = self._transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Pool each log to single vector (mean over sequence)
        if log_mask is not None:
            flat_mask = log_mask.view(-1, seq_len, 1).float()
            log_vectors = (hidden * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp(min=1)
        else:
            log_vectors = hidden.mean(dim=1)

        # Reshape
        log_vectors = log_vectors.view(batch_size, num_logs, -1)

        return log_vectors

    def pool(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool multiple logs using attention."""
        attn_scores = self._log_attention(hidden_states).squeeze(-1)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.einsum('bn,bnd->bd', attn_weights, hidden_states)

        return pooled
