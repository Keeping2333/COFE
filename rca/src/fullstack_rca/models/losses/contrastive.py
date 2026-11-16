"""
Contrastive Loss for 4-Layer Aware Training.

Implements contrastive learning objectives that:
1. Pull same-layer anomalies closer in embedding space
2. Push different-layer anomalies apart
3. Maintain separation between normal and anomalous samples

This enables the model to learn layer-discriminative representations
that support accurate root cause layer identification.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LayerContrastiveLoss(nn.Module):
    """
    Layer-aware contrastive loss for RCA.

    Uses InfoNCE-style loss where:
    - Positive pairs: samples from the SAME fault layer (e.g., L1 and L1)
    - Negative pairs: samples from DIFFERENT fault layers (e.g., L1 and L2)

    This encourages the model to learn representations where:
    - L1 anomalies cluster together (CPU, memory, disk issues)
    - L2 anomalies cluster together (network, latency issues)
    - L3 anomalies cluster together (pod, orchestration issues)
    - L4 anomalies cluster together (application, code issues)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        margin: float = 0.5,
    ) -> None:
        """
        Initialize layer contrastive loss.

        Args:
            temperature: Temperature parameter for softmax scaling.
                Lower = sharper distribution, harder to learn.
                Higher = softer distribution, easier but less discriminative.
            use_hard_negatives: Whether to use hard negative mining.
            margin: Margin for triplet-style loss component.
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.margin = margin

    def forward(
        self,
        embeddings: Tensor,
        layer_labels: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Compute layer-aware contrastive loss.

        Args:
            embeddings: [batch_size, embed_dim] - encoded representations
            layer_labels: [batch_size] - layer labels (0=L1, 1=L2, 2=L3, 3=L4)
                Use -1 for samples without labels (normal samples).
            valid_mask: Optional [batch_size] - which samples to include

        Returns:
            Tuple of (loss, metrics_dict):
            - loss: Scalar loss value
            - metrics_dict: Dictionary with diagnostic metrics
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Filter out samples without labels (label == -1)
        if valid_mask is None:
            valid_mask = layer_labels >= 0

        # Need at least 2 valid samples
        num_valid = valid_mask.sum()
        if num_valid < 2:
            return torch.tensor(0.0, device=device), {"num_valid": 0}

        # Get valid embeddings and labels
        valid_embeddings = embeddings[valid_mask]
        valid_labels = layer_labels[valid_mask]

        # Normalize embeddings for cosine similarity
        valid_embeddings = F.normalize(valid_embeddings, p=2, dim=1)

        # Compute pairwise similarity matrix
        # [num_valid, num_valid]
        sim_matrix = torch.mm(valid_embeddings, valid_embeddings.t())

        # Create masks for positive and negative pairs
        # Positive: same layer label
        label_eq_mask = valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)

        # Remove diagonal (self-similarity)
        eye_mask = torch.eye(num_valid, dtype=torch.bool, device=device)
        label_eq_mask = label_eq_mask & ~eye_mask

        # Negative: different layer label
        label_ne_mask = ~label_eq_mask & ~eye_mask

        # Apply temperature
        sim_matrix = sim_matrix / self.temperature

        # InfoNCE loss
        # For each sample, we want to maximize similarity with positives
        # and minimize similarity with negatives

        loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(num_valid):
            # Positive indices for sample i
            pos_mask = label_eq_mask[i]

            if pos_mask.sum() == 0:
                continue  # No positives for this sample

            # Negative indices for sample i
            neg_mask = label_ne_mask[i]

            if neg_mask.sum() == 0:
                continue  # No negatives for this sample

            # Positive similarities
            pos_sims = sim_matrix[i][pos_mask]

            # Negative similarities
            neg_sims = sim_matrix[i][neg_mask]

            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # For multiple positives, we average over them

            for pos_sim in pos_sims:
                # Denominator: this positive + all negatives
                if self.use_hard_negatives:
                    # Use hardest negative (highest similarity)
                    hard_neg_sim = neg_sims.max()
                    logits = torch.stack([pos_sim, hard_neg_sim])
                else:
                    # Use all negatives
                    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])

                # Cross-entropy loss (positive is index 0)
                labels = torch.zeros(1, dtype=torch.long, device=device)
                loss += F.cross_entropy(logits.unsqueeze(0), labels)
                num_pairs += 1

        if num_pairs > 0:
            loss = loss / num_pairs

        # Compute metrics
        metrics = {
            "num_valid": num_valid.item(),
            "num_pairs": num_pairs,
            "mean_pos_sim": sim_matrix[label_eq_mask].mean().item() if label_eq_mask.sum() > 0 else 0,
            "mean_neg_sim": sim_matrix[label_ne_mask].mean().item() if label_ne_mask.sum() > 0 else 0,
        }

        return loss, metrics


class HierarchicalContrastiveLoss(nn.Module):
    """
    Hierarchical contrastive loss that respects layer relationships.

    In addition to same/different layer contrast, this loss considers
    that some layers are more related than others:
    - L1 (Infrastructure) <-> L2 (Network): somewhat related
    - L3 (Platform) <-> L4 (Application): somewhat related
    - L1 <-> L4: very different

    This hierarchy is encoded as a distance matrix.
    """

    # Layer similarity matrix (higher = more similar)
    # L1, L2, L3, L4
    LAYER_SIMILARITY = torch.tensor([
        [1.0, 0.5, 0.2, 0.1],  # L1
        [0.5, 1.0, 0.3, 0.2],  # L2
        [0.2, 0.3, 1.0, 0.5],  # L3
        [0.1, 0.2, 0.5, 1.0],  # L4
    ])

    def __init__(
        self,
        temperature: float = 0.07,
        soft_labels: bool = True,
    ) -> None:
        """
        Initialize hierarchical contrastive loss.

        Args:
            temperature: Temperature for softmax scaling
            soft_labels: Whether to use soft labels based on layer similarity
        """
        super().__init__()
        self.temperature = temperature
        self.soft_labels = soft_labels
        self.register_buffer("layer_sim", self.LAYER_SIMILARITY)

    def forward(
        self,
        embeddings: Tensor,
        layer_labels: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Compute hierarchical contrastive loss.

        Args:
            embeddings: [batch_size, embed_dim]
            layer_labels: [batch_size] - layer indices (0-3)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Filter valid samples
        valid_mask = layer_labels >= 0
        num_valid = valid_mask.sum()

        if num_valid < 2:
            return torch.tensor(0.0, device=device), {"num_valid": 0}

        valid_embeddings = embeddings[valid_mask]
        valid_labels = layer_labels[valid_mask]

        # Normalize
        valid_embeddings = F.normalize(valid_embeddings, p=2, dim=1)

        # Compute similarity
        sim_matrix = torch.mm(valid_embeddings, valid_embeddings.t())
        sim_matrix = sim_matrix / self.temperature

        # Get target similarities from layer hierarchy
        # [num_valid, num_valid]
        target_sim = self.layer_sim.to(device)[valid_labels][:, valid_labels]

        # Remove diagonal
        eye_mask = torch.eye(num_valid, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(eye_mask, -1e9)

        if self.soft_labels:
            # KL divergence between predicted and target distributions
            pred_dist = F.softmax(sim_matrix, dim=-1)
            target_dist = F.softmax(target_sim / self.temperature, dim=-1)
            target_dist = target_dist.masked_fill(eye_mask, 0)
            target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            loss = F.kl_div(
                pred_dist.log(),
                target_dist,
                reduction='batchmean',
            )
        else:
            # Use target similarities as weights for contrastive loss
            weights = target_sim.masked_fill(eye_mask, 0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

            # Weighted cross-entropy
            log_probs = F.log_softmax(sim_matrix, dim=-1)
            loss = -(weights * log_probs).sum(dim=-1).mean()

        metrics = {
            "num_valid": num_valid.item(),
            "mean_sim": sim_matrix[~eye_mask].mean().item(),
        }

        return loss, metrics


class TripletMarginLoss(nn.Module):
    """
    Triplet loss for layer-aware learning.

    For each anchor sample, selects:
    - Positive: same layer sample
    - Negative: different layer sample

    Learns to satisfy: d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(
        self,
        margin: float = 0.5,
        mining: str = "hard",
    ) -> None:
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
            mining: Mining strategy ("hard", "semi-hard", "all")
        """
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(
        self,
        embeddings: Tensor,
        layer_labels: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Compute triplet loss.

        Args:
            embeddings: [batch_size, embed_dim]
            layer_labels: [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Filter valid
        valid_mask = layer_labels >= 0
        num_valid = valid_mask.sum()

        if num_valid < 3:
            return torch.tensor(0.0, device=device), {"num_triplets": 0}

        valid_embeddings = embeddings[valid_mask]
        valid_labels = layer_labels[valid_mask]

        # Compute pairwise distances
        dist_matrix = torch.cdist(valid_embeddings, valid_embeddings, p=2)

        # Mine triplets
        loss = torch.tensor(0.0, device=device)
        num_triplets = 0

        for anchor_idx in range(num_valid):
            anchor_label = valid_labels[anchor_idx]

            # Positive mask
            pos_mask = (valid_labels == anchor_label)
            pos_mask[anchor_idx] = False

            # Negative mask
            neg_mask = (valid_labels != anchor_label)

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # Distances
            pos_dists = dist_matrix[anchor_idx][pos_mask]
            neg_dists = dist_matrix[anchor_idx][neg_mask]

            if self.mining == "hard":
                # Hardest positive: furthest same-label
                hard_pos = pos_dists.max()
                # Hardest negative: closest different-label
                hard_neg = neg_dists.min()

                triplet_loss = F.relu(hard_pos - hard_neg + self.margin)
                loss += triplet_loss
                num_triplets += 1

            elif self.mining == "all":
                # All valid triplets
                for pos_dist in pos_dists:
                    for neg_dist in neg_dists:
                        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
                        if triplet_loss > 0:
                            loss += triplet_loss
                            num_triplets += 1

        if num_triplets > 0:
            loss = loss / num_triplets

        metrics = {
            "num_triplets": num_triplets,
            "mean_loss": loss.item() if num_triplets > 0 else 0,
        }

        return loss, metrics


class CombinedContrastiveLoss(nn.Module):
    """
    Combines multiple contrastive objectives.

    Default combination:
    - LayerContrastiveLoss: main layer discrimination
    - Classification loss: supervised layer prediction
    """

    def __init__(
        self,
        temperature: float = 0.07,
        classification_weight: float = 0.5,
        contrastive_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.contrastive_loss = LayerContrastiveLoss(temperature=temperature)
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        embeddings: Tensor,
        layer_logits: Tensor,
        layer_labels: Tensor,
    ) -> Tuple[Tensor, dict]:
        """
        Compute combined loss.

        Args:
            embeddings: [batch, embed_dim] - for contrastive
            layer_logits: [batch, 4] - for classification
            layer_labels: [batch] - ground truth layers

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Contrastive loss
        contrast_loss, contrast_metrics = self.contrastive_loss(embeddings, layer_labels)

        # Classification loss (only for valid labels)
        valid_mask = layer_labels >= 0
        if valid_mask.sum() > 0:
            class_loss = F.cross_entropy(
                layer_logits[valid_mask],
                layer_labels[valid_mask],
            )
        else:
            class_loss = torch.tensor(0.0, device=embeddings.device)

        # Combine
        total_loss = (
            self.contrastive_weight * contrast_loss +
            self.classification_weight * class_loss
        )

        metrics = {
            **contrast_metrics,
            "contrastive_loss": contrast_loss.item(),
            "classification_loss": class_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics


class FaultTypeContrastiveLoss(nn.Module):
    """
    Fault-type level contrastive loss with hierarchical layer awareness.

    Unlike LayerContrastiveLoss (4-class, causes within-layer collapse),
    this uses 12 fault types as the primary contrastive target:
    - Positive: same fault_type_id
    - Intra-layer negative: same layer, different fault_type (moderate push)
    - Inter-layer negative: different layer (strong push)

    Also includes a classification head loss for fault_type prediction.
    """

    # Layer distance matrix: higher = more dissimilar
    # L1(0), L2(1), L3(2), L4(3)
    LAYER_DISTANCE = torch.tensor([
        [0.0, 0.3, 0.7, 1.0],  # L1
        [0.3, 0.0, 0.5, 0.7],  # L2
        [0.7, 0.5, 0.0, 0.3],  # L3
        [1.0, 0.7, 0.3, 0.0],  # L4
    ])

    def __init__(
        self,
        temperature: float = 0.07,
        intra_layer_weight: float = 0.5,
        classification_weight: float = 0.3,
        num_fault_types: int = 12,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.classification_weight = classification_weight
        self.num_fault_types = num_fault_types
        self.intra_layer_weight = intra_layer_weight
        self.register_buffer("layer_dist", self.LAYER_DISTANCE)

    def forward(
        self,
        embeddings: Tensor,
        fault_type_ids: Tensor,
        layer_labels: Tensor,
        layer_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Args:
            embeddings: [B, D] - fused embeddings
            fault_type_ids: [B] - fault type labels (0-11), -1 for unknown
            layer_labels: [B] - layer labels (0-3), -1 for normal
            layer_logits: [B, 4] - optional layer classification logits
        """
        device = embeddings.device
        layer_dist = self.layer_dist.to(device)

        # Filter valid samples (both fault_type and layer must be known)
        valid_mask = (fault_type_ids >= 0) & (layer_labels >= 0)
        num_valid = valid_mask.sum()

        if num_valid < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "num_valid": 0, "contrastive_loss": 0.0,
            }

        emb = F.normalize(embeddings[valid_mask], p=2, dim=1)
        ft_ids = fault_type_ids[valid_mask]
        ly_ids = layer_labels[valid_mask]
        n = emb.shape[0]

        # Pairwise cosine similarity / temperature
        sim = torch.mm(emb, emb.t()) / self.temperature

        # Masks
        eye = torch.eye(n, dtype=torch.bool, device=device)
        same_ft = ft_ids.unsqueeze(0) == ft_ids.unsqueeze(1)  # positive
        same_ly = ly_ids.unsqueeze(0) == ly_ids.unsqueeze(1)
        intra_neg = same_ly & ~same_ft & ~eye  # same layer, different FT
        inter_neg = ~same_ly & ~eye             # different layer
        pos_mask = same_ft & ~eye

        # --- Contrastive loss ---
        # For each anchor, use all positives vs weighted negatives
        loss = torch.tensor(0.0, device=device)
        num_anchors = 0

        for i in range(n):
            if pos_mask[i].sum() == 0:
                continue

            pos_sims = sim[i][pos_mask[i]]

            # Collect negatives with hierarchical weighting
            neg_indices = intra_neg[i] | inter_neg[i]
            if neg_indices.sum() == 0:
                continue

            neg_sims = sim[i][neg_indices]

            # Weight inter-layer negatives by layer distance (harder push for distant layers)
            neg_layer_ids = ly_ids[neg_indices]
            anchor_layer = ly_ids[i]
            layer_weights = layer_dist[anchor_layer][neg_layer_ids]

            # Intra-layer negatives get intra_layer_weight
            is_intra = intra_neg[i][neg_indices]
            neg_weights = torch.where(
                is_intra,
                torch.full_like(layer_weights, self.intra_layer_weight),
                layer_weights.clamp(min=0.3),  # inter-layer: use distance, min 0.3
            )

            # Weighted InfoNCE: for each positive, compute loss against all negatives
            # Scale negative logits by weights to control push strength
            weighted_neg_sims = neg_sims + neg_weights.log()  # log-weight in logit space

            for pos_sim in pos_sims:
                logits = torch.cat([pos_sim.unsqueeze(0), weighted_neg_sims])
                target = torch.zeros(1, dtype=torch.long, device=device)
                loss += F.cross_entropy(logits.unsqueeze(0), target)
                num_anchors += 1

        if num_anchors > 0:
            loss = loss / num_anchors

        total_loss = loss

        # --- Optional classification loss ---
        cls_loss_val = 0.0
        if layer_logits is not None and self.classification_weight > 0:
            valid_logits = layer_logits[valid_mask]
            cls_loss = F.cross_entropy(valid_logits, ly_ids)
            cls_loss_val = cls_loss.item()
            total_loss = total_loss + self.classification_weight * cls_loss

        # Metrics
        with torch.no_grad():
            raw_sim = torch.mm(emb, emb.t())
            metrics = {
                "num_valid": num_valid.item(),
                "num_anchors": num_anchors,
                "contrastive_loss": loss.item(),
                "classification_loss": cls_loss_val,
                "mean_pos_sim": raw_sim[pos_mask].mean().item() if pos_mask.sum() > 0 else 0,
                "mean_intra_neg_sim": raw_sim[intra_neg].mean().item() if intra_neg.sum() > 0 else 0,
                "mean_inter_neg_sim": raw_sim[inter_neg].mean().item() if inter_neg.sum() > 0 else 0,
            }

        return total_loss, metrics
