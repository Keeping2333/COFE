#!/usr/bin/env python3
'English text removed.'
import argparse
import torch
from fullstack_rca.data.dataset import collate_multimodal_samples
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.fusion.projector import MultiModalProjector
from fullstack_rca.models.losses.contrastive import LayerContrastiveLoss
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Test fusion and contrastive learning")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset.yaml config (default: rca/config/dataset.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"English text removed.: {config['dataset']['type']} ({config['dataset']['path']})")

          
    dataset = create_dataset(config["dataset"])
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_multimodal_samples)
    batch = next(iter(loader))

         
    trace_encoder = TraceEncoder(
        span_dim=batch['span_features'].shape[-1],
        output_dim=256,
    ).to(device)

    log_encoder = LogEncoder(output_dim=256, freeze_bert=True).to(device)
    metric_encoder = create_metric_encoder(
        num_metrics=batch["metric_values"].shape[1],
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

          
    loss_fn = LayerContrastiveLoss(temperature=0.07)

          
    trace_emb = trace_encoder({
        'span_features': batch['span_features'].to(device),
        'trace_adjacency': batch['trace_adjacency'].to(device),
    }).embeddings

    log_emb = log_encoder({
        'log_tokens': batch['log_tokens'].to(device),
        'log_mask': batch['log_mask'].to(device),
    }).embeddings

    metric_emb, _ = metric_encoder(
        metric_values=batch["metric_values"].to(device),
        relative_time_positions=batch["relative_time_positions"].to(device),
    )

        
    fused = projector(log_emb, metric_emb, trace_emb)
    print(f"English text removed. embedding: {fused['fused'].shape}")

              
    labels = batch["fault_layers"].to(device)

    loss, metrics = loss_fn(fused['fused'], labels)
    print(f"English text removed.: {loss.item():.4f}")
    print(f"English text removed.: {metrics}")
    print('English text removed.')

if __name__ == "__main__":
    main()
