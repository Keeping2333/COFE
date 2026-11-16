#!/usr/bin/env python3
'English text removed.'
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

                                                                        
                                                   
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from fullstack_rca.data.dataset import (
    FAULT_TYPE_TO_ID,
    collate_multimodal_samples,
)
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.split import (
    experiment_level_split,
    get_one_trace_per_experiment,
)
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.fusion.projector import MultiModalProjector

                                                                          
FT_ORDER = [ft for ft, _ in sorted(FAULT_TYPE_TO_ID.items(), key=lambda kv: kv[1])]


def parse_args():
    p = argparse.ArgumentParser(description="KB geometry / τ design diagnostic")
    p.add_argument("--config", type=str, default=None,
                   help="Path to dataset.yaml (default: rca/config/dataset.yaml)")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_6span",
                   help='English text removed.')
    p.add_argument("--kb-dir", type=str, default="knowledge_base_6span_agg",
                   help="KB directory (default: experiment-level agg KB; "
                        "pass knowledge_base_6span for legacy trace-level)")
    p.add_argument("--ood-anomaly-dir", type=str,
                   default="train-ticket/exp_runs/round2_ood_anomaly",
                   help="9-experiment OOD-anomaly dataset root")
    p.add_argument("--ood-service-dir", type=str,
                   default="train-ticket/exp_runs/round2_ood_service",
                   help="24-experiment OOD-service dataset root (silent-injection unfiltered)")
    p.add_argument("--output-dir", type=str, default="results/kb_geometry_6span",
                   help="Where to write histograms + distributions.json")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def build_encoder_stack(ckpt_path: str, device: torch.device, sample_batch: dict):
    """Reconstruct the 6span encoder stack with trace_encoder_config from the ckpt."""
    span_dim = sample_batch["span_features"].shape[-1]
    _ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _ckpt.get("trace_encoder_config", {})

    trace_encoder = TraceEncoder(
        span_dim=int(cfg.get("span_dim", span_dim)),
        hidden_dim=128, output_dim=256, num_heads=4, num_gat_layers=2,
        enable_rich_features=bool(cfg.get("enable_rich_features", False)),
        num_service_buckets=int(cfg.get("num_service_buckets", 64)),
        num_operation_buckets=int(cfg.get("num_operation_buckets", 1024)),
        service_embed_dim=int(cfg.get("service_embed_dim", 16)),
        operation_embed_dim=int(cfg.get("operation_embed_dim", 32)),
        status_embed_dim=int(cfg.get("status_embed_dim", 4)),
        num_status_classes=int(cfg.get("num_status_classes", 4)),
        num_error_type_buckets=int(cfg.get("num_error_type_buckets", 32)),
        error_type_embed_dim=int(cfg.get("error_type_embed_dim", 8)),
        num_otel_status_classes=int(cfg.get("num_otel_status_classes", 3)),
        otel_status_embed_dim=int(cfg.get("otel_status_embed_dim", 4)),
    ).to(device)
    log_encoder = LogEncoder(
        output_dim=256, pretrained_model="distilbert-base-uncased",
        freeze_bert=True, unfreeze_last_n=0,
    ).to(device)
    metric_encoder = create_metric_encoder(
        num_metrics=sample_batch["metric_values"].shape[1],
        output_dim=256, initial_sigma=30.0,
    ).to(device)
    projector = MultiModalProjector(
        log_dim=256, trace_dim=256, metric_dim=256,
        output_dim=512, fusion_method="concat",
    ).to(device)

    trace_encoder.load_state_dict(_ckpt["trace_encoder"], strict=True)
    log_encoder.load_state_dict(_ckpt["log_encoder"], strict=True)
    metric_encoder.load_state_dict(_ckpt["metric_encoder"], strict=True)
    projector.load_state_dict(_ckpt["projector"], strict=True)
    for m in (trace_encoder, log_encoder, metric_encoder, projector):
        m.eval()
    return trace_encoder, log_encoder, metric_encoder, projector


def encode_subset(loader, encoders, device) -> np.ndarray:
    """Return [N, 512] L2-normalized fused embeddings for the loader."""
    trace_enc, log_enc, metric_enc, projector = encoders
    out: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            trace_inputs = {
                "span_features": batch["span_features"].to(device),
                "trace_adjacency": batch["trace_adjacency"].to(device),
            }
            for k in ("span_service_ids", "span_operation_ids",
                     "span_status_classes", "span_is_root", "span_is_leaf",
                     "span_is_on_critical_path"):
                if k in batch:
                    trace_inputs[k] = batch[k].to(device)
            trace_emb = trace_enc(trace_inputs).embeddings
            log_emb = log_enc({
                "log_tokens": batch["log_tokens"].to(device),
                "log_mask": batch["log_mask"].to(device),
            }).embeddings
            metric_emb, _ = metric_enc(
                metric_values=batch["metric_values"].to(device),
                relative_time_positions=batch["relative_time_positions"].to(device),
            )
            fused = projector(log_emb, metric_emb, trace_emb)["fused"]
            fused = torch.nn.functional.normalize(fused, p=2, dim=1)
            out.append(fused.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)


def fault_types_for_indices(dataset, indices: List[int]) -> List[str]:
    """Read root_cause_type from dataset._samples_index in the order of `indices`.

    The collate function does NOT carry root_cause strings into the batch,
    only fault_type_id (which is -1 for OOD-anomaly's brand-new fault types
    like connection_reset). So we go to the source-of-truth index instead.
    """
    return [dataset._samples_index[i]["root_cause"] for i in indices]


def load_dataset_with_override(base_config: dict, path: str):
    """Re-instantiate dataset with config["dataset"]["path"] overridden."""
    cfg = deepcopy(base_config)
    cfg["dataset"]["path"] = path
    return create_dataset(cfg["dataset"])


def all_one_trace_loader(dataset, batch_size: int, num_workers: int = 2, device=None):
    """Build a loader over ALL experiments in `dataset` (one trace each).

    Returns (loader, indices) — indices are the dataset positions, in the
    same order the loader yields, so callers can recover per-sample metadata
    (root_cause / run_id) via dataset._samples_index[i].
    """
    all_exps = sorted({s["run_id"] for s in dataset._samples_index})
    indices = get_one_trace_per_experiment(dataset._samples_index, all_exps)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multimodal_samples,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
    )
    return loader, indices


def percentiles(arr: np.ndarray, ps=(1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def summarize(name: str, arr: np.ndarray) -> dict:
    pct = percentiles(arr)
    return {
        "n": int(arr.size),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else float("nan"),
        **pct,
    }


def print_summary(label: str, x):
    """Accept either an ndarray (will summarize inline) or a pre-built summary dict."""
    s = x if isinstance(x, dict) else summarize(label, x)
    print(f"  {label:24s} n={s['n']:3d}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
          f"p1={s['p1']:+.4f}  p5={s['p5']:+.4f}  p50={s['p50']:+.4f}  "
          f"p95={s['p95']:+.4f}  p99={s['p99']:+.4f}")


def plot_heatmap(mat: np.ndarray, labels: List[str], out_path: str):
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(mat[i,j]) < 0.5 else "white")
    plt.colorbar(im, ax=ax, label="cosine sim")
    ax.set_title("Inter-FT centroid cosine sim (12×12)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_intra_ft(per_ft_sims: Dict[str, np.ndarray], out_path: str):
    if not HAS_MPL: return
    n_ft = len(per_ft_sims)
    cols = 4; rows = (n_ft + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.5*rows), sharex=True)
    axes = axes.flatten()
    for i, ft in enumerate(FT_ORDER):
        ax = axes[i]
        sims = per_ft_sims.get(ft, np.array([]))
        if sims.size:
            ax.hist(sims, bins=20, range=(0, 1), edgecolor="black")
            ax.axvline(np.percentile(sims, 5), color="red", linestyle="--",
                       label=f"p5={np.percentile(sims,5):.3f}")
            ax.legend(fontsize=7)
        ax.set_title(ft, fontsize=8)
        ax.set_xlim(0, 1)
    for j in range(len(FT_ORDER), len(axes)): axes[j].axis("off")
    fig.suptitle("Intra-FT episode → own-FT centroid cosine sim", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_loo_top1_per_ft(per_ft_sims: Dict[str, np.ndarray], out_path: str):
    """Overlay 12 FTs' LOO 1-NN sim distributions as step-line histograms.

    Step-line (histtype='step') keeps it readable even with 12 series.
    Each FT also gets a thin vertical at its p5 to make τ_FT candidates
    visually scannable.
    """
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = plt.get_cmap("tab20")
    for i, ft in enumerate(FT_ORDER):
        sims = per_ft_sims.get(ft)
        if sims is None or sims.size == 0:
            continue
        p5 = float(np.percentile(sims, 5))
        ax.hist(sims, bins=60, range=(0, 1), alpha=0.85,
                label=f"{ft} (p5={p5:.3f}, n={sims.size})",
                color=palette(i), histtype="step", linewidth=1.4)
        ax.axvline(p5, color=palette(i), linestyle=":", alpha=0.5, linewidth=0.9)
    ax.set_xlim(0.4, 1.005)
    ax.set_xlabel("LOO 1-NN cosine sim (within same FT)")
    ax.set_ylabel("count")
    ax.set_title("Per-FT LOO 1-NN sim (12 overlay) — informs τ_FT for V2 per-cluster detection")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_top1_overlay(id_sims: np.ndarray,
                      ood_anom_sims: np.ndarray, ood_anom_fts: List[str],
                      ood_svc_sims: np.ndarray, out_path: str):
    if not HAS_MPL: return
    fig, ax = plt.subplots(figsize=(10, 5))
    if id_sims.size:
        ax.hist(id_sims, bins=30, range=(0, 1), alpha=0.55, label=f"ID test (n={id_sims.size})",
                color="steelblue", edgecolor="black")
    if ood_svc_sims.size:
        ax.hist(ood_svc_sims, bins=30, range=(0, 1), alpha=0.55,
                label=f"OOD-service unfiltered (n={ood_svc_sims.size})",
                color="goldenrod", edgecolor="black")
    if ood_anom_sims.size:
                                              
        unique_fts = sorted(set(ood_anom_fts))
        palette = ["crimson", "darkmagenta", "darkgreen"]
        for ft, color in zip(unique_fts, palette):
            mask = np.array([f == ft for f in ood_anom_fts])
            ax.scatter(ood_anom_sims[mask], np.full(mask.sum(), -1.0),
                       color=color, s=80, marker="v", label=f"OOD-anom: {ft} (n={mask.sum()})",
                       edgecolor="black", linewidth=0.6, zorder=5)
    ax.set_xlim(0, 1); ax.set_xlabel("max cosine sim against 285-episode KB (unfiltered)")
    ax.set_ylabel("count (histograms); OOD-anom shown as markers at y=-1")
    ax.set_title("Top-1 sim distribution: ID vs OOD")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"KB dir:         {args.kb_dir}")

                                                                  
                                                      
                                                                        
                                                                       
                                                                        
                                                                        
                                                                  
    print("\n[1/6] Loading KB and computing FT centroids...")
    faiss_index = faiss.read_index(os.path.join(args.kb_dir, "index.faiss"))
    n_kb = int(faiss_index.ntotal)
    kb_vectors = faiss_index.reconstruct_n(0, n_kb).astype(np.float32)
                                                                          
    norms = np.linalg.norm(kb_vectors, axis=1, keepdims=True)
    kb_vectors = kb_vectors / np.maximum(norms, 1e-9)

    with open(os.path.join(args.kb_dir, "metadata.json")) as f:
        meta = json.load(f)
    id_list = meta["id_list"]
    meta_map = meta["metadata"]
    if len(id_list) != n_kb:
        raise RuntimeError(f"id_list len {len(id_list)} != FAISS ntotal {n_kb}")
    kb_fts = [meta_map.get(i, {}).get("fault_type", "unknown") for i in id_list]
    print(f"  KB size: {n_kb} vectors  |  {len(set(kb_fts))} FTs present")

                                                                                         
    centroids: Dict[str, np.ndarray] = {}
    per_ft_intra: Dict[str, np.ndarray] = {}
    for ft in FT_ORDER:
        mask = np.array([f == ft for f in kb_fts])
        if not mask.any():
            print(f"  WARNING: FT '{ft}' has 0 episodes in KB — skipping")
            continue
        ft_vecs = kb_vectors[mask]
        c = ft_vecs.mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-9)
        centroids[ft] = c
        per_ft_intra[ft] = (ft_vecs @ c)                                  
        print(f"    {ft:35s}  n={int(mask.sum()):3d}  "
              f"intra p5={np.percentile(per_ft_intra[ft],5):.4f}  "
              f"p50={np.percentile(per_ft_intra[ft],50):.4f}  "
              f"min={per_ft_intra[ft].min():.4f}")

                                                                                           
    inter_mat = np.full((len(FT_ORDER), len(FT_ORDER)), np.nan, dtype=np.float32)
    for i, fi in enumerate(FT_ORDER):
        for j, fj in enumerate(FT_ORDER):
            if fi in centroids and fj in centroids:
                inter_mat[i, j] = float(centroids[fi] @ centroids[fj])

    plot_heatmap(inter_mat, FT_ORDER, os.path.join(args.output_dir, "heatmap_ft_centroids.png"))
    plot_intra_ft(per_ft_intra, os.path.join(args.output_dir, "hist_intra_ft.png"))

                                                                  
                                                                          
                                                                  
    print("\n[2/6] Building dataset loaders and encoders...")
                                                                              
                                                                 
    id_dataset = create_dataset(config["dataset"])
    train_exps, test_exps = experiment_level_split(
        id_dataset._samples_index, test_ratio=0.2, seed=42,
    )
    test_indices = get_one_trace_per_experiment(id_dataset._samples_index, test_exps)
    id_test_loader = DataLoader(
        torch.utils.data.Subset(id_dataset, test_indices),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_multimodal_samples, num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    sample_batch = next(iter(id_test_loader))
    encoders = build_encoder_stack(os.path.join(args.ckpt_dir, "best_model.pt"),
                                   device, sample_batch)
    print(f"  ID test:        {len(test_indices)} experiments  "
          f"(experiment_level_split, test_ratio=0.2, seed=42)")

                                                                                                
    ood_anom_ds = load_dataset_with_override(config, args.ood_anomaly_dir)
    ood_anom_loader, ood_anom_idx = all_one_trace_loader(
        ood_anom_ds, args.batch_size, device=device)
    print(f"  OOD-anomaly:    {len(ood_anom_idx)} experiments")

                                                                      
    ood_svc_ds = load_dataset_with_override(config, args.ood_service_dir)
    ood_svc_loader, ood_svc_idx = all_one_trace_loader(
        ood_svc_ds, args.batch_size, device=device)
    print(f"  OOD-service:    {len(ood_svc_idx)} experiments  "
          " — silent-injection NOT filtered (per progress.md §05.17 ~9-15 may be real)")

                                                                  
                                                             
                                                                  
    print("\n[3/6] Encoding ID-test...")
    id_test_emb = encode_subset(id_test_loader, encoders, device)
    print("[3/6] Encoding OOD-anomaly...")
    ood_anom_emb = encode_subset(ood_anom_loader, encoders, device)
    ood_anom_fts = fault_types_for_indices(ood_anom_ds, ood_anom_idx)
    print("[3/6] Encoding OOD-service...")
    ood_svc_emb = encode_subset(ood_svc_loader, encoders, device)
    ood_svc_fts = fault_types_for_indices(ood_svc_ds, ood_svc_idx)

    def top1_against_kb(emb: np.ndarray) -> np.ndarray:
        if emb.size == 0:
            return np.array([])
                                                                        
        sims = emb @ kb_vectors.T             
        return sims.max(axis=1)

    id_top1 = top1_against_kb(id_test_emb)
    ood_anom_top1 = top1_against_kb(ood_anom_emb)
    ood_svc_top1 = top1_against_kb(ood_svc_emb)

    plot_top1_overlay(id_top1, ood_anom_top1, ood_anom_fts, ood_svc_top1,
                      os.path.join(args.output_dir, "hist_top1_overlay.png"))

                                                                  
                                                 
                                                                  
    print("\n[4/6] Distribution summaries (cosine sim, all positive direction):")
    print("\n  ── Inter-FT centroid sim (off-diagonal) ─────────────────────────")
    off = inter_mat[np.triu_indices(len(FT_ORDER), k=1)]
    off = off[~np.isnan(off)]
    print_summary("inter-FT (off-diag)", off)

    print("\n  ── Intra-FT episode → own-centroid sim, per FT ──────────────────")
    intra_summary = {}
    for ft in FT_ORDER:
        if ft in per_ft_intra:
            s = summarize(ft, per_ft_intra[ft])
            intra_summary[ft] = s
            print_summary(ft, per_ft_intra[ft])

    print("\n  ── Top-1 sim against KB (unfiltered global) ─────────────────────")
    id_summary = summarize("ID-test", id_top1)
    ood_anom_summary = summarize("OOD-anomaly", ood_anom_top1)
    ood_svc_summary = summarize("OOD-service-raw", ood_svc_top1)
    print_summary("ID-test (72)", id_top1)
    print_summary("OOD-anomaly (9)", ood_anom_top1)
    print_summary("OOD-service (24, raw)", ood_svc_top1)

                                  
    print("\n  ── OOD-anomaly per-FT breakdown ─────────────────────────────────")
    ood_anom_per_ft = {}
    for ft in sorted(set(ood_anom_fts)):
        mask = np.array([f == ft for f in ood_anom_fts])
        ood_anom_per_ft[ft] = summarize(ft, ood_anom_top1[mask])
        print_summary(f"  {ft}", ood_anom_top1[mask])

    out_json = {
        "ckpt_dir": args.ckpt_dir,
        "kb_dir": args.kb_dir,
        "kb_size": int(kb_vectors.shape[0]),
        "inter_ft_centroid_sim_matrix": inter_mat.tolist(),
        "inter_ft_offdiag_summary": summarize("off_diag", off),
        "ft_order": FT_ORDER,
        "intra_ft_summary": intra_summary,
        "id_test": {
            "summary": id_summary,
            "values": id_top1.tolist(),
        },
        "ood_anomaly": {
            "summary": ood_anom_summary,
            "values": ood_anom_top1.tolist(),
            "fault_types": ood_anom_fts,
            "per_ft_summary": ood_anom_per_ft,
        },
        "ood_service_raw": {
            "summary": ood_svc_summary,
            "values": ood_svc_top1.tolist(),
            "fault_types": ood_svc_fts,
            "note": "silent-injection NOT filtered; see progress.md §05.17",
        },
    }

                                                                  
                                                                   
                                                                  
                                                                     
                                                                      
                                                                  
    print("\n[5/6] Per-FT LOO 1-NN within same FT...")
    per_ft_loo_top1: Dict[str, np.ndarray] = {}
    for ft in FT_ORDER:
        mask = np.array([f == ft for f in kb_fts])
        if mask.sum() < 2:
            print(f"  SKIP {ft} (n={int(mask.sum())} — need ≥2 for LOO)")
            continue
        ft_vecs = kb_vectors[mask]                                      
        sims = ft_vecs @ ft_vecs.T                              
        np.fill_diagonal(sims, -np.inf)                                   
        per_ft_loo_top1[ft] = sims.max(axis=1).astype(np.float32)
        del sims, ft_vecs                                                     

    print("\n  ── Per-FT LOO 1-NN sim (intra-cluster) ──────────────────────────")
    per_ft_loo_summary = {}
    for ft in FT_ORDER:
        if ft in per_ft_loo_top1:
            per_ft_loo_summary[ft] = summarize(ft, per_ft_loo_top1[ft])
            print_summary(ft, per_ft_loo_top1[ft])

    out_json["per_ft_loo_top1_summary"] = per_ft_loo_summary

    plot_loo_top1_per_ft(per_ft_loo_top1,
                         os.path.join(args.output_dir, "hist_loo_top1_per_ft.png"))

                                                                  
                                           
                                                                  
    json_path = os.path.join(args.output_dir, "distributions.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"\n[6/6] Wrote:")
    print(f"  {os.path.join(args.output_dir, 'heatmap_ft_centroids.png')}")
    print(f"  {os.path.join(args.output_dir, 'hist_intra_ft.png')}")
    print(f"  {os.path.join(args.output_dir, 'hist_top1_overlay.png')}")
    print(f"  {os.path.join(args.output_dir, 'hist_loo_top1_per_ft.png')}")
    print(f"  {json_path}")
    if not HAS_MPL:
        print("  (matplotlib not installed — only JSON written; pip install matplotlib for plots)")


if __name__ == "__main__":
    main()
