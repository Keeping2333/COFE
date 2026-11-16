#!/usr/bin/env python3
'English text removed.'
import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from fullstack_rca.data.alignment import build_trace_graph, preprocess_jaeger_spans
from fullstack_rca.data.dataset import collate_multimodal_samples
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.data.sampler import StratifiedFaultTypeSampler
from fullstack_rca.data.split import experiment_level_split, get_experiment_indices
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.fusion.projector import MultiModalProjector
from fullstack_rca.models.losses.contrastive import FaultTypeContrastiveLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Contrastive Learning Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset.yaml config (default: rca/config/dataset.yaml)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint_dir/best_model.pt (load weights and continue from next epoch)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20). Use a smaller value for quick test runs.")
    return parser.parse_args()


def compute_duration_stats(dataset, indices):
    """Compute log1p(duration) statistics from RAW span data.

    CRITICAL: This reads raw span durations (microseconds) directly from disk,
    bypassing dataset.__getitem__() which applies per-trace max normalization.
    Using normalized values would produce wrong z-score stats.

    Args:
        dataset: ExpRunsDataset or MicroServiceDataset instance
        indices: list of sample indices to compute over (training set only)

    Returns:
        dict with 'mean' and 'std' of log1p(raw_duration_us)
    """
    all_log_durs = []

    for idx in tqdm(indices, desc="Computing duration stats"):
        info = dataset._samples_index[idx]

        if hasattr(dataset, '_load_trace'):
                                                      
            run_id = info["run_id"]
            _, raw_spans = dataset._load_trace(run_id, info["trace_idx"])
            raw_spans = raw_spans[:dataset.max_spans_per_trace]
            spans = preprocess_jaeger_spans(raw_spans)
        else:
                                                             
            scenario = info["scenario_name"]
            trace = dataset._trace_cache[scenario][info["trace_idx"]]
            spans = trace.get("spans", [])[:dataset.max_spans_per_trace]

        if not spans:
            continue

                                                         
        raw_durations = torch.tensor(
            [float(s.get("duration", 0)) for s in spans],
            dtype=torch.float32,
        )
        all_log_durs.append(torch.log1p(raw_durations))

    if not all_log_durs:
        print("  WARNING: No spans found for duration stats, using defaults")
        return {"mean": 0.0, "std": 1.0}

    all_log_durs = torch.cat(all_log_durs)
    stats = {"mean": float(all_log_durs.mean()), "std": float(all_log_durs.std())}
    print(f"  Duration stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"  Computed from {len(all_log_durs)} spans across {len(indices)} samples")
    return stats


def main():
    args = parse_args()
    config = load_config(args.config)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

                                                      
                                                    
    batch_size = 8
    num_workers = 2
    num_epochs = args.epochs

    print("\n=== Training Configuration ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
    print(f"Dataset: {config['dataset']['type']} ({config['dataset']['path']})")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print("==============================\n")

                         
    print('English text removed.')
    dataset = create_dataset(config["dataset"])

                                                          
    train_exps, val_exps = experiment_level_split(dataset._samples_index, test_ratio=0.2, seed=42)
    train_indices = get_experiment_indices(dataset._samples_index, train_exps)
    val_indices = get_experiment_indices(dataset._samples_index, val_exps)

                                                                                                 
    duration_stats = compute_duration_stats(dataset, train_indices)
    dataset.duration_stats = duration_stats

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

                                                                   
                                                                    
                                                             
    train_ft_ids = []
    train_layers = []
    for local_idx in range(len(train_dataset)):
        sample = train_dataset[local_idx]
        train_ft_ids.append(sample.fault_type_id)
        train_layers.append(sample.fault_layer)

    train_sampler = StratifiedFaultTypeSampler(
        fault_type_ids=train_ft_ids,
        fault_layers=train_layers,
        batch_size=batch_size,
        drop_last=False,
    )
    print(f"  Stratified sampler: {len(train_sampler)} batches from {len(train_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_multimodal_samples,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,                                               
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_samples,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    print(f"  English text removed.: {len(train_dataset)}, English text removed.: {len(val_dataset)}")

                          
    print('English text removed.')

            
    sample_batch = next(iter(train_loader))
    span_dim = sample_batch['span_features'].shape[-1]

                                                                              
                                                                          
                                                                     
                                                                        
                               
    model_cfg = config.get("model", {}).get("trace", {})
    dataset_params = config.get("dataset", {}).get("params", {})
    trace_enable_rich = bool(
        model_cfg.get("enable_rich_features", False)
        or dataset_params.get("enable_rich_span_features", False)
    )

    trace_encoder = TraceEncoder(
        span_dim=span_dim,
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_gat_layers=2,
        enable_rich_features=trace_enable_rich,
        num_service_buckets=int(dataset_params.get("num_service_buckets", 64)),
        num_operation_buckets=int(dataset_params.get("num_operation_buckets", 1024)),
        service_embed_dim=int(model_cfg.get("service_embed_dim", 16)),
        operation_embed_dim=int(model_cfg.get("operation_embed_dim", 32)),
        status_embed_dim=int(model_cfg.get("status_embed_dim", 4)),
        num_status_classes=int(model_cfg.get("num_status_classes", 4)),
        num_error_type_buckets=int(dataset_params.get("num_error_type_buckets", 32)),
        error_type_embed_dim=int(model_cfg.get("error_type_embed_dim", 8)),
        num_otel_status_classes=int(model_cfg.get("num_otel_status_classes", 3)),
        otel_status_embed_dim=int(model_cfg.get("otel_status_embed_dim", 4)),
    ).to(device)

    log_encoder = LogEncoder(
        output_dim=256,
        pretrained_model="distilbert-base-uncased",
        freeze_bert=True,                        
        unfreeze_last_n=0,                           
    ).to(device)

    num_metrics = int(sample_batch["metric_values"].shape[1])
    metric_encoder = create_metric_encoder(
        num_metrics=num_metrics,
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

    loss_fn = FaultTypeContrastiveLoss(
        temperature=0.07,
        intra_layer_weight=0.5,
        classification_weight=0.3,
    )

                                
                                                  
    bert_params = []
    non_bert_params = []
    for name, param in log_encoder.named_parameters():
        if param.requires_grad:
            if '_bert.' in name:
                bert_params.append(param)
            else:
                non_bert_params.append(param)

    param_groups = [
        {"params": bert_params, "lr": 1e-5},                  
        {"params": non_bert_params
         + list(trace_encoder.parameters())
         + list(metric_encoder.parameters())
         + list(projector.parameters()),
         "lr": 1e-4},                                     
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    bert_trainable = sum(p.numel() for p in bert_params)
    total_trainable = sum(p.numel() for g in param_groups for p in g["params"] if p.requires_grad)
    print(f"  BERT unfrozen params: {bert_trainable:,} / Total trainable: {total_trainable:,}")

                         
    print('English text removed.')
    print('English text removed.')
    print(f"  English text removed.: {config['dataset']['type']} ({len(dataset)} English text removed.)")

    best_val_loss = float('inf')
    best_epoch = -1
    start_epoch = 0

    if args.resume:
        ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
        resume_path = os.path.join(checkpoint_dir, 'resume_state.pt')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            trace_encoder.load_state_dict(ckpt['trace_encoder'], strict=True)
            log_encoder.load_state_dict(ckpt['log_encoder'], strict=True)
            metric_encoder.load_state_dict(ckpt['metric_encoder'], strict=True)
            projector.load_state_dict(ckpt['projector'], strict=True)
            print(f"  Loaded weights from {ckpt_path}")
            if os.path.isfile(resume_path):
                rs = torch.load(resume_path, map_location=device, weights_only=False)
                start_epoch = rs.get('epoch', -1) + 1
                best_val_loss = rs.get('best_val_loss', float('inf'))
                best_epoch = rs.get('best_epoch', -1)
                print(f"  Resume from epoch {start_epoch} (best so far: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            else:
                start_epoch = 0
                print("  No resume_state.pt found, starting from epoch 0")
        else:
            print("  No best_model.pt found, starting from scratch")

    if start_epoch >= num_epochs:
        print(f"\n  Already completed {num_epochs} epochs (resume_state says next epoch={start_epoch}). Nothing to do.")
        return

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

            
        trace_encoder.train()
        log_encoder.train()
        metric_encoder.train()
        projector.train()

        train_loss = 0.0
        train_loss_sum = 0.0
        train_total_pairs = 0
        train_total_valid = 0
        train_effective_batches = 0
        train_steps = 0
        train_samples = 0
        train_pos_sim_sum = 0.0
        train_intra_neg_sim_sum = 0.0
        train_inter_neg_sim_sum = 0.0
        train_sim_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            batch_size_actual = batch["span_features"].shape[0]
            train_samples += batch_size_actual

                
            trace_inputs = {
                'span_features': batch['span_features'].to(device),
                'trace_adjacency': batch['trace_adjacency'].to(device),
            }
            for k in (
                'span_service_ids', 'span_operation_ids',
                'span_status_classes', 'span_is_root', 'span_is_leaf',
                'span_is_on_critical_path',
            ):
                if k in batch:
                    trace_inputs[k] = batch[k].to(device)
            trace_emb = trace_encoder(trace_inputs).embeddings

            log_emb = log_encoder({
                'log_tokens': batch['log_tokens'].to(device),
                'log_mask': batch['log_mask'].to(device),
            }).embeddings

            metric_emb, _ = metric_encoder(
                metric_values=batch["metric_values"].to(device),
                relative_time_positions=batch["relative_time_positions"].to(device),
            )

                
            fused = projector(log_emb, metric_emb, trace_emb)

                
            labels = batch["fault_layers"].to(device)

                
            ft_ids = batch["fault_type_ids"].to(device)
            loss, metrics = loss_fn(
                fused['fused'], ft_ids, labels,
                layer_logits=fused.get('layer_logits'),
            )
            batch_pairs = int(metrics.get("num_anchors", 0))
            train_total_pairs += batch_pairs
            train_total_valid += int(metrics.get("num_valid", 0))

                                                                   
            if batch_pairs > 0:
                train_pos_sim_sum += float(metrics.get("mean_pos_sim", 0.0))
                train_intra_neg_sim_sum += float(metrics.get("mean_intra_neg_sim", 0.0))
                train_inter_neg_sim_sum += float(metrics.get("mean_inter_neg_sim", 0.0))
                train_sim_batches += 1

                                        
            if batch_pairs > 0:
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                train_effective_batches += 1
                train_steps += 1

        train_loss = train_loss_sum / max(1, train_effective_batches)
        train_pos_sim = train_pos_sim_sum / max(1, train_sim_batches)
        train_intra_neg_sim = train_intra_neg_sim_sum / max(1, train_sim_batches)
        train_inter_neg_sim = train_inter_neg_sim_sum / max(1, train_sim_batches)

            
        trace_encoder.eval()
        log_encoder.eval()
        metric_encoder.eval()
        projector.eval()

        val_loss_sum = 0.0
        val_total_pairs = 0
        val_total_valid = 0
        val_effective_batches = 0
        val_samples = 0
        val_pos_sim_sum = 0.0
        val_intra_neg_sim_sum = 0.0
        val_inter_neg_sim_sum = 0.0
        val_sim_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_size_actual = batch["span_features"].shape[0]
                val_samples += batch_size_actual

                trace_inputs = {
                    'span_features': batch['span_features'].to(device),
                    'trace_adjacency': batch['trace_adjacency'].to(device),
                }
                for k in (
                    'span_service_ids', 'span_operation_ids',
                    'span_status_classes', 'span_is_root', 'span_is_leaf',
                'span_is_on_critical_path',
                ):
                    if k in batch:
                        trace_inputs[k] = batch[k].to(device)
                trace_emb = trace_encoder(trace_inputs).embeddings

                log_emb = log_encoder({
                    'log_tokens': batch['log_tokens'].to(device),
                    'log_mask': batch['log_mask'].to(device),
                }).embeddings

                metric_emb, _ = metric_encoder(
                    metric_values=batch["metric_values"].to(device),
                    relative_time_positions=batch["relative_time_positions"].to(device),
                )
                fused = projector(log_emb, metric_emb, trace_emb)

                labels = batch["fault_layers"].to(device)

                ft_ids = batch["fault_type_ids"].to(device)
                loss, metrics = loss_fn(
                    fused['fused'], ft_ids, labels,
                    layer_logits=fused.get('layer_logits'),
                )
                batch_pairs = int(metrics.get("num_anchors", 0))
                val_total_pairs += batch_pairs
                val_total_valid += int(metrics.get("num_valid", 0))

                if batch_pairs > 0:
                    val_loss_sum += loss.item()
                    val_effective_batches += 1
                    val_pos_sim_sum += float(metrics.get("mean_pos_sim", 0.0))
                    val_intra_neg_sim_sum += float(metrics.get("mean_intra_neg_sim", 0.0))
                    val_inter_neg_sim_sum += float(metrics.get("mean_inter_neg_sim", 0.0))
                    val_sim_batches += 1

        val_loss = val_loss_sum / max(1, val_effective_batches)
        val_pos_sim = val_pos_sim_sum / max(1, val_sim_batches)
        val_intra_neg_sim = val_intra_neg_sim_sum / max(1, val_sim_batches)
        val_inter_neg_sim = val_inter_neg_sim_sum / max(1, val_sim_batches)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Train pairs/batch: {train_total_pairs / max(1, len(train_loader)):.2f} (total={train_total_pairs})")
        print(f"  Val pairs/batch:   {val_total_pairs / max(1, len(val_loader)):.2f} (total={val_total_pairs})")
        print(f"  Train mean_pos_sim: {train_pos_sim:.4f}")
        print(f"  Train mean_intra_neg_sim: {train_intra_neg_sim:.4f}")
        print(f"  Train mean_inter_neg_sim: {train_inter_neg_sim:.4f}")
        print(f"  Train sim_gap (pos-intra): {(train_pos_sim - train_intra_neg_sim):.4f}")
        print(f"  Train sim_gap (pos-inter): {(train_pos_sim - train_inter_neg_sim):.4f}")
        print(f"  Val mean_pos_sim:   {val_pos_sim:.4f}")
        print(f"  Val mean_intra_neg_sim:   {val_intra_neg_sim:.4f}")
        print(f"  Val mean_inter_neg_sim:   {val_inter_neg_sim:.4f}")
        print(f"  Val sim_gap (pos-intra): {(val_pos_sim - val_intra_neg_sim):.4f}")
        print(f"  Val sim_gap (pos-inter): {(val_pos_sim - val_inter_neg_sim):.4f}")
        print(f"  Effective train batches: {train_effective_batches}/{len(train_loader)}")
        print(f"  Effective val batches:   {val_effective_batches}/{len(val_loader)}")
        print(f"  Valid labels seen (train/val): {train_total_valid}/{val_total_valid}")
        print(f"  Optimizer steps: {train_steps}")
        print(f"  Samples processed (train/val): {train_samples}/{val_samples}")
        print(f"  Learning rate: {current_lr:.6f}")
        print(f"  Epoch time: {epoch_time:.1f}s")
        if val_effective_batches == 0:
            print("  WARNING: No effective validation batches with valid contrastive pairs.")
            print("           Validation loss is not reliable for checkpoint selection.")

                
        if val_effective_batches > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_dict = {
                'trace_encoder': trace_encoder.state_dict(),
                'log_encoder': log_encoder.state_dict(),
                'metric_encoder': metric_encoder.state_dict(),
                'projector': projector.state_dict(),
                'duration_stats': duration_stats,
                                                                             
                                                                             
                                                                          
                                
                'num_metrics': num_metrics,
                                                                                
                                                                 
                'trace_encoder_config': {
                    'span_dim': span_dim,
                    'enable_rich_features': trace_enable_rich,
                    'num_service_buckets': int(dataset_params.get('num_service_buckets', 64)),
                    'num_operation_buckets': int(dataset_params.get('num_operation_buckets', 1024)),
                    'service_embed_dim': int(model_cfg.get('service_embed_dim', 16)),
                    'operation_embed_dim': int(model_cfg.get('operation_embed_dim', 32)),
                    'status_embed_dim': int(model_cfg.get('status_embed_dim', 4)),
                    'num_status_classes': int(model_cfg.get('num_status_classes', 4)),
                                                                                     
                    'num_error_type_buckets': int(dataset_params.get('num_error_type_buckets', 32)),
                    'error_type_embed_dim': int(model_cfg.get('error_type_embed_dim', 8)),
                    'num_otel_status_classes': int(model_cfg.get('num_otel_status_classes', 3)),
                    'otel_status_embed_dim': int(model_cfg.get('otel_status_embed_dim', 4)),
                },
            }
            torch.save(ckpt_dict, os.path.join(checkpoint_dir, 'best_model.pt'))
            torch.save({
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, os.path.join(checkpoint_dir, 'resume_state.pt'))
            print(f"  New best checkpoint saved (val_loss={val_loss:.4f})")

    print('English text removed.')
    if best_epoch > 0:
        print(f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    else:
        print("  No checkpoint selected from validation (no effective validation batches).")
        print("  Please use retrieval metrics and/or improve validation batch composition.")

if __name__ == "__main__":
    _args = parse_args()
    _cfg = load_config(_args.config)
    os.makedirs(_cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    main()
