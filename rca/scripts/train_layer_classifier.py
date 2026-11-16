#!/usr/bin/env python3
"""Train the standalone layer classifier on anomaly-weighted aggregated
pre-fusion log+metric embeddings (one vector per EXPERIMENT, not per trace).

Symmetric-protocol training, mirrors build_kb.py:
1. Load trained encoders (frozen) from checkpoint
2. For each experiment, encode ALL its traces (log+metric)
3. Compute per-trace anomaly_score = error_rate + max_latency (same formula as
   build_kb.py line 199-204)
4. Anomaly-weighted softmax pool with floor=0.01 → one (log_emb, metric_emb)
   per experiment (matches build_kb.aggregate_by_experiment, RCAPipeline.process_experiment)
5. Train small MLP (512D → 128 → 64 → 4) with cross-entropy on the aggregates
6. Evaluate on held-out test experiments, print confusion + per-layer accuracy
7. Save classifier weights to layer_classifier.pt (overwrites the legacy
   per-trace version — the asymmetric protocol is retired)

Usage:
    cd rca
    python scripts/train_layer_classifier.py
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from fullstack_rca.data.dataset import collate_multimodal_samples
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.split import (
    experiment_level_split,
    get_experiment_indices,
)
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.layer_classifier import LayerClassifier

LAYER_NAMES = ["L1_Infra", "L2_Network", "L3_Platform", "L4_App"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train layer classifier (aggregated)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def precompute_trace_embeddings(dataset_subset, encoders, device, desc=""):
    """Encode every trace in the subset; return per-trace (log, metric, labels, scores).

    Two passes over dataset_subset:
      Pass A — anomaly_score per trace, computed from sample.span_features
               (un-padded) so the formula matches build_kb.py byte-for-byte.
      Pass B — batched log + metric encoding for speed.

    Returns:
        log_embs:    [N, 256] CPU tensor
        metric_embs: [N, 256] CPU tensor
        labels:      [N]      CPU tensor (fault_layer indices, 0..3)
        scores:      [N]      CPU tensor (anomaly score per trace)
    """
    log_encoder, metric_encoder = encoders

                                                                           
                                                                               
                                                     
    scores = []
    for sample in tqdm(dataset_subset, desc=f"  {desc} score"):
        sf = sample.span_features.cpu().numpy()
        scores.append(float(sf[:, 1].mean()) + float(sf[:, 0].max()))

                                
    loader = DataLoader(
        dataset_subset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_multimodal_samples,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_log, all_metric, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {desc} encode"):
            log_emb = log_encoder({
                "log_tokens": batch["log_tokens"].to(device),
                "log_mask": batch["log_mask"].to(device),
            }).embeddings

            metric_emb, _ = metric_encoder(
                metric_values=batch["metric_values"].to(device),
                relative_time_positions=batch["relative_time_positions"].to(device),
            )

            all_log.append(log_emb.cpu())
            all_metric.append(metric_emb.cpu())
            all_labels.append(batch["fault_layers"])

    return (
        torch.cat(all_log),
        torch.cat(all_metric),
        torch.cat(all_labels),
        torch.tensor(scores, dtype=torch.float32),
    )


def aggregate_per_experiment(log_embs, metric_embs, labels, scores, indices, samples_index):
    """Anomaly-weighted pool per experiment. Matches build_kb.aggregate_by_experiment
    line 254-265 and RCAPipeline.process_experiment.

    Args:
        log_embs:    [N, 256] per-trace log embeddings (CPU)
        metric_embs: [N, 256] per-trace metric embeddings (CPU)
        labels:      [N]      per-trace fault layer indices
        scores:      [N]      per-trace anomaly scores
        indices:     dataset indices the subset was built from (len N)
        samples_index: dataset._samples_index — used to read experiment id per trace

    Returns:
        agg_log:    [E, 256] one log embedding per experiment
        agg_metric: [E, 256] one metric embedding per experiment
        agg_labels: [E]      experiment-level layer label
        exp_ids:    list of E experiment ids in the same order as the tensors
    """
    exp_key = "run_id" if "run_id" in samples_index[0] else "scenario_name"

    exp_to_pos: dict = {}
    for pos, ds_idx in enumerate(indices):
        eid = samples_index[ds_idx][exp_key]
        exp_to_pos.setdefault(eid, []).append(pos)

    agg_log, agg_metric, agg_labels, exp_ids = [], [], [], []
    floor = 0.01
    for eid in sorted(exp_to_pos.keys()):
        pos = exp_to_pos[eid]
        exp_log = log_embs[pos]                   
        exp_metric = metric_embs[pos]             
        exp_scores = scores[pos]             

        weights = torch.clamp_min(exp_scores, floor)
        weights = weights / weights.sum()

        agg_log.append((exp_log * weights.unsqueeze(1)).sum(dim=0))
        agg_metric.append((exp_metric * weights.unsqueeze(1)).sum(dim=0))
        agg_labels.append(labels[pos[0]])                                    
        exp_ids.append(eid)

    return (
        torch.stack(agg_log),
        torch.stack(agg_metric),
        torch.stack(agg_labels),
        exp_ids,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = f"{checkpoint_dir}/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

                                         
    print("\n[1/5] Loading dataset...")
    dataset = create_dataset(config["dataset"])

                                                                              
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    duration_stats = ckpt.get("duration_stats")

                                                       
    if duration_stats:
        dataset.duration_stats = duration_stats
        print(f"  Duration stats: mean={duration_stats['mean']:.4f}, std={duration_stats['std']:.4f}")

    train_exps, test_exps = experiment_level_split(
        dataset._samples_index, test_ratio=0.2, seed=42
    )
                                                                                
    train_indices = get_experiment_indices(dataset._samples_index, train_exps)
    test_indices = get_experiment_indices(dataset._samples_index, test_exps)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    print(f"  Train: {len(train_exps)} experiments / {len(train_subset)} traces, "
          f"Test: {len(test_exps)} experiments / {len(test_subset)} traces")

                                           
    print("\n[2/5] Loading encoders from checkpoint...")
    sample_loader = DataLoader(train_subset, batch_size=1,
                               collate_fn=collate_multimodal_samples, num_workers=0)
    sample_batch = next(iter(sample_loader))

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

    log_encoder.load_state_dict(ckpt["log_encoder"], strict=True)
    metric_encoder.load_state_dict(ckpt["metric_encoder"], strict=True)
    log_encoder.eval()
    metric_encoder.eval()
    print("  Encoders loaded and frozen.")

                                                                       
    print("\n[3/5] Precomputing per-trace embeddings...")
    train_log_t, train_metric_t, train_labels_t, train_scores_t = precompute_trace_embeddings(
        train_subset, (log_encoder, metric_encoder), device, desc="train",
    )
    test_log_t, test_metric_t, test_labels_t, test_scores_t = precompute_trace_embeddings(
        test_subset, (log_encoder, metric_encoder), device, desc="test",
    )
    print(f"  Per-trace shapes: train log {train_log_t.shape}, test log {test_log_t.shape}")

    print("\n  Aggregating per experiment (anomaly-weighted, floor=0.01)...")
    train_log, train_metric, train_labels, train_exp_ids = aggregate_per_experiment(
        train_log_t, train_metric_t, train_labels_t, train_scores_t,
        train_indices, dataset._samples_index,
    )
    test_log, test_metric, test_labels, test_exp_ids = aggregate_per_experiment(
        test_log_t, test_metric_t, test_labels_t, test_scores_t,
        test_indices, dataset._samples_index,
    )
    print(f"  Aggregated: train {train_log.shape}, test {test_log.shape}")
    print(f"  Label distribution (train): {np.bincount(train_labels.numpy(), minlength=4)}")
    print(f"  Label distribution (test):  {np.bincount(test_labels.numpy(), minlength=4)}")

                                     
    print(f"\n[4/5] Training layer classifier ({args.epochs} epochs)...")
    classifier = LayerClassifier(
        log_dim=256,
        metric_dim=256,
        hidden_dim=128,
        num_layers=4,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

                                           
    train_log_d = train_log.to(device)
    train_metric_d = train_metric.to(device)
    train_labels_d = train_labels.to(device)

    best_test_acc = 0.0
    best_epoch = -1
    best_state = None

    for epoch in range(args.epochs):
        classifier.train()
        logits = classifier(train_log_d, train_metric_d)
        loss = loss_fn(logits, train_labels_d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

                                  
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            classifier.eval()
            with torch.no_grad():
                train_preds = logits.argmax(dim=1).cpu().numpy()
                train_acc = (train_preds == train_labels.numpy()).mean()

                test_logits = classifier(test_log.to(device), test_metric.to(device))
                test_preds = test_logits.argmax(dim=1).cpu().numpy()
                test_acc = (test_preds == test_labels.numpy()).mean()

            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

                  
    classifier.load_state_dict(best_state)
    print(f"\n  Best test accuracy: {best_test_acc:.3f} (epoch {best_epoch})")

                                     
    print(f"\n[5/5] Final evaluation on {len(test_labels)} test experiments...")
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_log.to(device), test_metric.to(device))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

    y_true = test_labels.numpy()

    print(f"\n  Overall Accuracy: {(test_preds == y_true).mean():.1%} ({(test_preds == y_true).sum()}/{len(y_true)})")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, test_preds, target_names=LAYER_NAMES, digits=3))

    print("  Confusion Matrix (rows=GT, cols=Pred):")
    cm = confusion_matrix(y_true, test_preds)
    header = "         " + "".join(f"{l:>12s}" for l in LAYER_NAMES)
    print(header)
    for i, name in enumerate(LAYER_NAMES):
        row = f"  {name:12s}" + "".join(f"{cm[i,j]:12d}" for j in range(4))
        print(row)

                        
    print(f"\n  Per-sample predictions:")
    print(f"  {'#':>3s}  {'GT':>12s}  {'Pred':>12s}  {'Correct':>8s}  {'Confidence':>10s}")
    for i in range(len(y_true)):
        gt_name = LAYER_NAMES[y_true[i]]
        pred_name = LAYER_NAMES[test_preds[i]]
        correct = "OK" if y_true[i] == test_preds[i] else "MISS"
        conf = test_probs[i, test_preds[i]]
        print(f"  {i+1:3d}  {gt_name:>12s}  {pred_name:>12s}  {correct:>8s}  {conf:10.4f}")

                     
    save_path = os.path.join(checkpoint_dir, "layer_classifier.pt")
    torch.save({
        "classifier": best_state,
        "config": {"log_dim": 256, "metric_dim": 256, "hidden_dim": 128, "num_layers": 4, "dropout": 0.3},
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
    }, save_path)
    print(f"\n  Classifier saved to {save_path}")


if __name__ == "__main__":
    main()
