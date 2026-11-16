#!/usr/bin/env python3
"""Per-modality centroid analysis: which modality best separates which layer pairs?

Loads the trained checkpoint, encodes training samples through each encoder
separately, computes per-layer centroids, and outputs 4×4 cosine similarity
matrices for each modality (log, metric, trace) plus the fused embedding.

Usage:
    cd rca
    python scripts/analyze_centroids.py
    python scripts/analyze_centroids.py --config config/dataset.yaml
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from fullstack_rca.data.dataset import collate_multimodal_samples
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.split import (
    experiment_level_split,
    get_one_trace_per_experiment,
)
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.fusion.projector import MultiModalProjector

LAYER_NAMES = ["L1", "L2", "L3", "L4"]


def parse_args():
    parser = argparse.ArgumentParser(description="Per-modality centroid analysis")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset.yaml config")
    return parser.parse_args()


def cosine_similarity_matrix(centroids: dict[str, np.ndarray]) -> np.ndarray:
    """Compute 4×4 cosine similarity matrix from layer centroids.

    Args:
        centroids: dict mapping layer name (L1-L4) to centroid vector.

    Returns:
        4×4 numpy array of cosine similarities.
    """
    mat = np.zeros((4, 4))
    for i, li in enumerate(LAYER_NAMES):
        for j, lj in enumerate(LAYER_NAMES):
            a, b = centroids[li], centroids[lj]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a < 1e-9 or norm_b < 1e-9:
                mat[i, j] = 0.0
            else:
                mat[i, j] = float(np.dot(a, b) / (norm_a * norm_b))
    return mat


def print_similarity_matrix(name: str, dim: int, mat: np.ndarray):
    """Pretty-print a 4×4 cosine similarity matrix."""
    print(f"\n{'=' * 60}")
    print(f"  {name} ({dim}D) — Layer Centroid Cosine Similarity")
    print(f"{'=' * 60}")
    header = "        " + "".join(f"{l:>8s}" for l in LAYER_NAMES)
    print(header)
    for i, li in enumerate(LAYER_NAMES):
        row = f"  {li}   "
        for j in range(4):
            row += f"{mat[i, j]:8.4f}"
        print(row)

                                               
    print(f"\n  Off-diagonal pairs (lower = better separation):")
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            sim = mat[i, j]
            pairs.append((LAYER_NAMES[i], LAYER_NAMES[j], sim))
            print(f"    {LAYER_NAMES[i]}↔{LAYER_NAMES[j]}: {sim:.4f}")
    return pairs


def main():
    args = parse_args()
    config = load_config(args.config)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = f"{checkpoint_dir}/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

                                                      
    print("\n[1/4] Loading dataset...")
    dataset = create_dataset(config["dataset"])
    train_exps, _test_exps = experiment_level_split(
        dataset._samples_index, test_ratio=0.2, seed=42
    )
    train_indices = get_one_trace_per_experiment(dataset._samples_index, train_exps)
    train_subset = torch.utils.data.Subset(dataset, train_indices)

    print(f"  Using {len(train_subset)} training experiments (one trace per experiment)")

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_multimodal_samples,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

                                      
    print("\n[2/4] Initializing models...")
    sample_batch = next(iter(train_loader))
    span_dim = sample_batch["span_features"].shape[-1]

                                                                                               
    _ckpt_for_cfg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _trace_cfg = _ckpt_for_cfg.get("trace_encoder_config", {})
    del _ckpt_for_cfg
    trace_encoder = TraceEncoder(
        span_dim=int(_trace_cfg.get("span_dim", span_dim)),
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_gat_layers=2,
        enable_rich_features=bool(_trace_cfg.get("enable_rich_features", False)),
        num_service_buckets=int(_trace_cfg.get("num_service_buckets", 64)),
        num_operation_buckets=int(_trace_cfg.get("num_operation_buckets", 1024)),
        service_embed_dim=int(_trace_cfg.get("service_embed_dim", 16)),
        operation_embed_dim=int(_trace_cfg.get("operation_embed_dim", 32)),
        status_embed_dim=int(_trace_cfg.get("status_embed_dim", 4)),
        num_status_classes=int(_trace_cfg.get("num_status_classes", 4)),
        num_error_type_buckets=int(_trace_cfg.get("num_error_type_buckets", 32)),
        error_type_embed_dim=int(_trace_cfg.get("error_type_embed_dim", 8)),
        num_otel_status_classes=int(_trace_cfg.get("num_otel_status_classes", 3)),
        otel_status_embed_dim=int(_trace_cfg.get("otel_status_embed_dim", 4)),
    ).to(device)

    log_encoder = LogEncoder(
        output_dim=256,
        pretrained_model="distilbert-base-uncased",
        freeze_bert=True,
        unfreeze_last_n=0,
    ).to(device)

    metric_encoder = create_metric_encoder(
        num_metrics=sample_batch["metric_values"].shape[1],
        output_dim=256,
        initial_sigma=30.0,
    ).to(device)

    projector = MultiModalProjector(
        log_dim=256,
        trace_dim=256,
        metric_dim=256,
        output_dim=512,
        fusion_method="concat",
    ).to(device)

                     
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trace_encoder.load_state_dict(ckpt["trace_encoder"], strict=True)
    log_encoder.load_state_dict(ckpt["log_encoder"], strict=True)
    metric_encoder.load_state_dict(ckpt["metric_encoder"], strict=True)
    projector.load_state_dict(ckpt["projector"], strict=True)

    trace_encoder.eval()
    log_encoder.eval()
    metric_encoder.eval()
    projector.eval()
    print("  Checkpoint loaded successfully.")

                                                
    print("\n[3/4] Encoding training samples...")

                                                           
    embeddings = {
        "log": {l: [] for l in LAYER_NAMES},
        "metric": {l: [] for l in LAYER_NAMES},
        "trace": {l: [] for l in LAYER_NAMES},
        "fused": {l: [] for l in LAYER_NAMES},
    }

    with torch.no_grad():
        for batch in train_loader:
                                  
            trace_inputs = {
                "span_features": batch["span_features"].to(device),
                "trace_adjacency": batch["trace_adjacency"].to(device),
            }
            for k in (
                "span_service_ids", "span_operation_ids",
                "span_status_classes", "span_is_root", "span_is_leaf",
                "span_is_on_critical_path",
            ):
                if k in batch:
                    trace_inputs[k] = batch[k].to(device)
            trace_emb = trace_encoder(trace_inputs).embeddings            

            log_emb = log_encoder({
                "log_tokens": batch["log_tokens"].to(device),
                "log_mask": batch["log_mask"].to(device),
            }).embeddings            

            metric_emb, _ = metric_encoder(
                metric_values=batch["metric_values"].to(device),
                relative_time_positions=batch["relative_time_positions"].to(device),
            )            

            fused = projector(log_emb, metric_emb, trace_emb)
            fused_emb = fused["fused"]            

                            
            layers = batch["fault_layers"].numpy()                                     
            for i in range(len(layers)):
                layer_idx = int(layers[i])
                layer_name = LAYER_NAMES[layer_idx]
                embeddings["log"][layer_name].append(log_emb[i].cpu().numpy())
                embeddings["metric"][layer_name].append(metric_emb[i].cpu().numpy())
                embeddings["trace"][layer_name].append(trace_emb[i].cpu().numpy())
                embeddings["fused"][layer_name].append(fused_emb[i].cpu().numpy())

                                   
    print("\n  Samples per layer:")
    for l in LAYER_NAMES:
        n = len(embeddings["log"][l])
        print(f"    {l}: {n} experiments")

                                                              
    print("\n[4/4] Computing centroids and cosine similarity matrices...")

    modality_dims = {"log": 256, "metric": 256, "trace": 256, "fused": 512}
    all_pairs = {}                                    

    for modality in ["log", "metric", "trace", "fused"]:
        centroids = {}
        for l in LAYER_NAMES:
            vecs = np.array(embeddings[modality][l])
            centroids[l] = vecs.mean(axis=0)

        mat = cosine_similarity_matrix(centroids)
        pairs = print_similarity_matrix(modality.capitalize(), modality_dims[modality], mat)
        all_pairs[modality] = {(a, b): sim for a, b, sim in pairs}

                                                               
    print(f"\n{'=' * 60}")
    print("  SUMMARY: Best Discriminator per Layer Pair")
    print(f"  (Lower cosine similarity = better separation)")
    print(f"{'=' * 60}")

    pair_keys = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair_keys.append((LAYER_NAMES[i], LAYER_NAMES[j]))

    for li, lj in pair_keys:
        modality_sims = []
        for mod in ["log", "metric", "trace"]:
            sim = all_pairs[mod][(li, lj)]
            modality_sims.append((mod, sim))
                                                                
        modality_sims.sort(key=lambda x: x[1])

        best_mod, best_sim = modality_sims[0]
        second_mod, second_sim = modality_sims[1]
        worst_mod, worst_sim = modality_sims[2]
        fused_sim = all_pairs["fused"][(li, lj)]

        print(f"\n  {li}↔{lj}:")
        print(f"    Best:   {best_mod:8s} (cos={best_sim:.4f})")
        print(f"    2nd:    {second_mod:8s} (cos={second_sim:.4f})  gap={second_sim - best_sim:+.4f}")
        print(f"    Worst:  {worst_mod:8s} (cos={worst_sim:.4f})  gap={worst_sim - best_sim:+.4f}")
        print(f"    Fused:  {'fused':8s} (cos={fused_sim:.4f})")

                                                    
    print(f"\n{'=' * 60}")
    print("  Embedding Norm Statistics (no L2 normalization)")
    print(f"{'=' * 60}")
    for modality in ["log", "metric", "trace"]:
        all_vecs = []
        for l in LAYER_NAMES:
            all_vecs.extend(embeddings[modality][l])
        norms = np.linalg.norm(all_vecs, axis=1)
        print(f"  {modality:8s}: mean={norms.mean():.2f}, std={norms.std():.2f}, "
              f"min={norms.min():.2f}, max={norms.max():.2f}")


if __name__ == "__main__":
    main()
