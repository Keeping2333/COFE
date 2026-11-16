#!/usr/bin/env python3
"""OOD detection evaluation: per-FT τ_FT vs pooled global τ.

Offline retrieval-only validator for the OOD gate. Does NOT call any LLM —
that lives in agent/ and is invoked by diagnose.py. The purpose here is to
verify whether the per-FT τ_FT decision rule cleanly separates ID test
samples from OOD-anomaly samples at the retrieval stage, before deciding
how to integrate the gate into RCAPipeline.

Decision rule:
  Encode sample → top-1 cosine sim against KB → look up FT of nearest KB
  vector → compare sim against τ_{nearest_ft}:
      sim ≥ τ → predict ID (with that FT)
      sim <  τ → predict OOD

Two τ regimes are evaluated in parallel, both derived from the same
percentile (default p5) of the KB's per-FT LOO 1-NN sim distribution
(written by analyze_kb_geometry.py to distributions.json):

  per-FT τ_FT : one threshold per cluster (12 values, heterogeneous)
  pooled τ    : single threshold over all per-FT LOO 1-NN sims combined

Outputs:
  results/eval_ood.json — per-sample records + 2×2 confusion matrices
  stdout                — confusion matrices + per-OOD-FT detection rates

Usage:
    conda activate fullstack_rca
    python rca/eval/eval_ood.py
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OOD detection evaluation (retrieval-only)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to dataset.yaml (default: rca/config/dataset.yaml)")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_6span")
    p.add_argument("--kb-dir", type=str, default="knowledge_base_6span_agg")
    p.add_argument("--ood-anomaly-dir", type=str,
                   default="train-ticket/exp_runs/round2_ood_anomaly")
    p.add_argument("--tau-source", type=str,
                   default="results/kb_geometry_6span/distributions.json",
                   help="JSON file produced by analyze_kb_geometry.py — supplies "
                        "per_ft_loo_top1_summary used to derive τ_FT and pooled τ")
    p.add_argument("--tau-percentile", type=int, default=5,
                   help="Percentile of per-FT LOO 1-NN sim used as τ "
                        "(default 5 → bottom 5%% of intra-cluster sim sets the floor)")
    p.add_argument("--output", type=str, default="results/eval_ood.json")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Encoder stack (mirrors analyze_kb_geometry.py to stay byte-aligned with
# the 6span checkpoint contract)
# ---------------------------------------------------------------------------
def build_encoder_stack(ckpt_path: str, device: torch.device, sample_batch: dict):
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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_dataset_with_override(base_config: dict, path: str):
    cfg = deepcopy(base_config)
    cfg["dataset"]["path"] = path
    return create_dataset(cfg["dataset"])


def all_one_trace_loader(dataset, batch_size: int, num_workers: int = 2,
                         pin_memory: bool = False) -> Tuple[DataLoader, List[int]]:
    all_exps = sorted({s["run_id"] for s in dataset._samples_index})
    indices = get_one_trace_per_experiment(dataset._samples_index, all_exps)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multimodal_samples,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return loader, indices


def sample_meta_for_indices(dataset, indices: List[int]) -> List[Dict[str, str]]:
    """Return [{run_id, fault_type, layer}, ...] in loader order."""
    out = []
    for i in indices:
        rec = dataset._samples_index[i]
        out.append({
            "run_id": rec["run_id"],
            "fault_type": rec["root_cause"],
            "layer": rec.get("layer", "unknown"),
        })
    return out


# ---------------------------------------------------------------------------
# τ derivation + decision
# ---------------------------------------------------------------------------
def derive_thresholds(tau_source: str, percentile: int) -> Tuple[Dict[str, float], float]:
    """Read per-FT LOO 1-NN summary and return (τ_FT dict, pooled τ).

    The JSON's per_ft_loo_top1_summary[ft] already contains percentile keys
    (p1/p5/p25/p50/p75/p95/p99), so per-FT τ is a direct lookup. The pooled
    τ requires raw values which the summary does not retain, so we
    approximate by pooling the per-FT p-th percentile values themselves and
    taking that scalar's percentile — equivalent in spirit, and within the
    same percentile-of-LOO-1-NN family as τ_FT.
    """
    with open(tau_source) as f:
        dist = json.load(f)
    per_ft_summary = dist["per_ft_loo_top1_summary"]
    key = f"p{percentile}"

    tau_ft: Dict[str, float] = {}
    for ft, s in per_ft_summary.items():
        tau_ft[ft] = float(s[key])

    # Pool: treat all FT's chosen-percentile values as a single distribution,
    # take the same percentile of that. Avoids privileging any one cluster.
    pooled_values = sorted(tau_ft.values())
    pooled_tau = float(np.percentile(pooled_values, percentile))
    return tau_ft, pooled_tau


def classify_sample(sim: float, nearest_ft: str,
                    tau_ft: Dict[str, float], tau_global: float
                    ) -> Tuple[str, str, float]:
    """Apply both decision rules. Returns (per_ft_decision, global_decision, tau_ft_used)."""
    tau = tau_ft.get(nearest_ft, tau_global)  # fallback if KB has no such FT
    per_ft = "ID" if sim >= tau else "OOD"
    glob = "ID" if sim >= tau_global else "OOD"
    return per_ft, glob, tau


# ---------------------------------------------------------------------------
# Confusion matrix + reporting
# ---------------------------------------------------------------------------
def confusion(samples: List[dict], decision_key: str) -> Dict[str, int]:
    tn = fp = fn = tp = 0
    for s in samples:
        gt = s["gt_class"]
        pred = s[decision_key]
        if gt == "ID" and pred == "ID":   tn += 1
        elif gt == "ID" and pred == "OOD": fp += 1
        elif gt == "OOD" and pred == "ID": fn += 1
        elif gt == "OOD" and pred == "OOD": tp += 1
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def rates(c: Dict[str, int]) -> Dict[str, float]:
    n_id = c["TN"] + c["FP"]
    n_ood = c["FN"] + c["TP"]
    return {
        "ood_recall":   c["TP"] / n_ood if n_ood else float("nan"),
        "id_retention": c["TN"] / n_id if n_id else float("nan"),
        "ood_precision": c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else float("nan"),
    }


def per_ood_ft_breakdown(samples: List[dict], decision_key: str) -> Dict[str, dict]:
    """Detection rate broken down by ground-truth OOD fault type."""
    by_ft: Dict[str, List[dict]] = {}
    for s in samples:
        if s["gt_class"] != "OOD":
            continue
        by_ft.setdefault(s["gt_ft"], []).append(s)
    out = {}
    for ft, items in by_ft.items():
        detected = sum(1 for s in items if s[decision_key] == "OOD")
        out[ft] = {"n": len(items), "detected": detected,
                   "rate": detected / len(items) if items else float("nan")}
    return out


def ft_accuracy_when_accepted(samples: List[dict], decision_key: str) -> Dict[str, float]:
    """Among samples we accept as ID, does retrieval top-1 give the correct FT?

    Computed only over gt_class=ID samples that the gate accepted (decision=ID).
    Mirrors diagnose.py's FT Acc but limited to retrieval-only (no LLM/SOP).
    """
    accepted = [s for s in samples
                if s["gt_class"] == "ID" and s[decision_key] == "ID"]
    if not accepted:
        return {"n": 0, "correct": 0, "accuracy": float("nan")}
    correct = sum(1 for s in accepted if s["nearest_ft"] == s["gt_ft"])
    return {"n": len(accepted), "correct": correct, "accuracy": correct / len(accepted)}


def print_confusion(title: str, c: Dict[str, int], r: Dict[str, float]):
    print(f"\n  {title}")
    print(f"  {'─' * 56}")
    print(f"                actual=ID    actual=OOD")
    print(f"  predicted=ID   {c['TN']:>4d} (TN)   {c['FN']:>4d} (FN)")
    print(f"  predicted=OOD  {c['FP']:>4d} (FP)   {c['TP']:>4d} (TP)")
    print(f"    OOD recall    : {r['ood_recall']:6.1%}  ({c['TP']}/{c['TP']+c['FN']})")
    print(f"    ID retention  : {r['id_retention']:6.1%}  ({c['TN']}/{c['TN']+c['FP']})")
    print(f"    OOD precision : {r['ood_precision']:6.1%}  "
          f"({c['TP']}/{c['TP']+c['FP'] if (c['TP']+c['FP']) else 0})")


def print_per_ood_breakdown(title: str, breakdown: Dict[str, dict]):
    if not breakdown:
        return
    print(f"\n  {title}")
    print(f"  {'─' * 56}")
    for ft, b in sorted(breakdown.items()):
        print(f"    {ft:32s}  {b['detected']:>2d}/{b['n']:<2d} = {b['rate']:6.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:         {device}")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"KB dir:         {args.kb_dir}")
    print(f"τ source:       {args.tau_source}  (percentile = p{args.tau_percentile})")

    # ------------------------------------------------------------------
    # (1) Thresholds
    # ------------------------------------------------------------------
    tau_ft, tau_global = derive_thresholds(args.tau_source, args.tau_percentile)
    print(f"\nPer-FT τ_FT (p{args.tau_percentile} of LOO 1-NN within each cluster):")
    for ft in FT_ORDER:
        if ft in tau_ft:
            print(f"    {ft:32s}  τ = {tau_ft[ft]:.4f}")
    print(f"Pooled global τ: {tau_global:.4f}")

    # ------------------------------------------------------------------
    # (2) KB metadata for nearest-FT lookup
    # ------------------------------------------------------------------
    faiss_index = faiss.read_index(os.path.join(args.kb_dir, "index.faiss"))
    n_kb = int(faiss_index.ntotal)
    kb_vectors = faiss_index.reconstruct_n(0, n_kb).astype(np.float32)
    kb_vectors = kb_vectors / np.maximum(
        np.linalg.norm(kb_vectors, axis=1, keepdims=True), 1e-9)
    with open(os.path.join(args.kb_dir, "metadata.json")) as f:
        meta = json.load(f)
    id_list = meta["id_list"]
    meta_map = meta["metadata"]
    kb_fts = [meta_map.get(i, {}).get("fault_type", "unknown") for i in id_list]
    print(f"\nKB: {n_kb} vectors, {len(set(kb_fts))} FTs present")

    # ------------------------------------------------------------------
    # (3) Build loaders + encoders
    # ------------------------------------------------------------------
    print("\nLoading ID test split (same seed=42 as diagnose.py)...")
    id_dataset = create_dataset(config["dataset"])
    _, test_exps = experiment_level_split(
        id_dataset._samples_index, test_ratio=0.2, seed=42)
    id_indices = get_one_trace_per_experiment(id_dataset._samples_index, test_exps)
    id_loader = DataLoader(
        torch.utils.data.Subset(id_dataset, id_indices),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_multimodal_samples, num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    sample_batch = next(iter(id_loader))
    encoders = build_encoder_stack(
        os.path.join(args.ckpt_dir, "best_model.pt"), device, sample_batch)
    id_meta = sample_meta_for_indices(id_dataset, id_indices)
    print(f"  {len(id_indices)} ID test experiments")

    print("Loading OOD-anomaly dataset...")
    ood_ds = load_dataset_with_override(config, args.ood_anomaly_dir)
    ood_loader, ood_indices = all_one_trace_loader(
        ood_ds, args.batch_size, pin_memory=(device.type == "cuda"))
    ood_meta = sample_meta_for_indices(ood_ds, ood_indices)
    print(f"  {len(ood_indices)} OOD-anomaly experiments")

    # ------------------------------------------------------------------
    # (4) Encode + top-1 retrieval
    # ------------------------------------------------------------------
    print("\nEncoding ID test...")
    id_emb = encode_subset(id_loader, encoders, device)
    print("Encoding OOD anomaly...")
    ood_emb = encode_subset(ood_loader, encoders, device)

    def top1(emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sims = emb @ kb_vectors.T  # [N, n_kb]
        idx = sims.argmax(axis=1)
        return sims[np.arange(len(idx)), idx], idx

    id_sim, id_nn_idx = top1(id_emb)
    ood_sim, ood_nn_idx = top1(ood_emb)

    # ------------------------------------------------------------------
    # (5) Build per-sample records + apply decisions
    # ------------------------------------------------------------------
    samples: List[dict] = []
    for k, (sim, nn_idx) in enumerate(zip(id_sim, id_nn_idx)):
        nearest_ft = kb_fts[int(nn_idx)]
        per_ft_dec, glob_dec, tau_used = classify_sample(
            float(sim), nearest_ft, tau_ft, tau_global)
        samples.append({
            "source": "id_test",
            "sample_id": id_meta[k]["run_id"],
            "gt_class": "ID",
            "gt_ft": id_meta[k]["fault_type"],
            "gt_layer": id_meta[k]["layer"],
            "nearest_sim": float(sim),
            "nearest_ft": nearest_ft,
            "tau_ft_used": tau_used,
            "per_ft_decision": per_ft_dec,
            "global_decision": glob_dec,
        })
    for k, (sim, nn_idx) in enumerate(zip(ood_sim, ood_nn_idx)):
        nearest_ft = kb_fts[int(nn_idx)]
        per_ft_dec, glob_dec, tau_used = classify_sample(
            float(sim), nearest_ft, tau_ft, tau_global)
        samples.append({
            "source": "ood_anomaly",
            "sample_id": ood_meta[k]["run_id"],
            "gt_class": "OOD",
            "gt_ft": ood_meta[k]["fault_type"],
            "gt_layer": ood_meta[k]["layer"],
            "nearest_sim": float(sim),
            "nearest_ft": nearest_ft,
            "tau_ft_used": tau_used,
            "per_ft_decision": per_ft_dec,
            "global_decision": glob_dec,
        })

    # ------------------------------------------------------------------
    # (6) Summaries
    # ------------------------------------------------------------------
    per_ft_conf = confusion(samples, "per_ft_decision")
    per_ft_rates = rates(per_ft_conf)
    per_ft_breakdown = per_ood_ft_breakdown(samples, "per_ft_decision")
    per_ft_id_acc = ft_accuracy_when_accepted(samples, "per_ft_decision")

    global_conf = confusion(samples, "global_decision")
    global_rates_d = rates(global_conf)
    global_breakdown = per_ood_ft_breakdown(samples, "global_decision")
    global_id_acc = ft_accuracy_when_accepted(samples, "global_decision")

    print("\n" + "=" * 60)
    print("OOD Detection Evaluation Results")
    print("=" * 60)
    print_confusion("Per-FT τ_FT decision rule", per_ft_conf, per_ft_rates)
    print(f"    FT Acc (when accepted): "
          f"{per_ft_id_acc['accuracy']:6.1%}  "
          f"({per_ft_id_acc['correct']}/{per_ft_id_acc['n']})")
    print_per_ood_breakdown("Per OOD type (per-FT τ)", per_ft_breakdown)

    print_confusion(f"Pooled global τ = {tau_global:.4f}",
                    global_conf, global_rates_d)
    print(f"    FT Acc (when accepted): "
          f"{global_id_acc['accuracy']:6.1%}  "
          f"({global_id_acc['correct']}/{global_id_acc['n']})")
    print_per_ood_breakdown(f"Per OOD type (global τ={tau_global:.4f})",
                            global_breakdown)

    # ------------------------------------------------------------------
    # (7) Persist JSON
    # ------------------------------------------------------------------
    output = {
        "config": {
            "checkpoint_dir": args.ckpt_dir,
            "kb_dir": args.kb_dir,
            "ood_anomaly_dir": args.ood_anomaly_dir,
            "tau_source": args.tau_source,
            "tau_percentile": args.tau_percentile,
        },
        "kb_size": n_kb,
        "thresholds": {
            "per_ft": tau_ft,
            "global": tau_global,
        },
        "samples": samples,
        "summary": {
            "per_ft": {
                "confusion": per_ft_conf,
                "rates": per_ft_rates,
                "per_ood_ft": per_ft_breakdown,
                "ft_accuracy_when_accepted": per_ft_id_acc,
            },
            "global": {
                "tau": tau_global,
                "confusion": global_conf,
                "rates": global_rates_d,
                "per_ood_ft": global_breakdown,
                "ft_accuracy_when_accepted": global_id_acc,
            },
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
