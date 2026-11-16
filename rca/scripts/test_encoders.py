#!/usr/bin/env python3
'English text removed.'
import argparse
import torch
from fullstack_rca.data.dataset import collate_multimodal_samples
from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.encoders.gaussian_attention import GaussianPositionalEncoding
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Test encoders")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset.yaml config (default: rca/config/dataset.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"English text removed.: {device}")
    print(f"English text removed.: {config['dataset']['type']} ({config['dataset']['path']})")

             
    dataset = create_dataset(config["dataset"])
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_multimodal_samples)
    batch = next(iter(loader))

                               
    print("\n=== Trace Encoder (GAT) ===")
    trace_encoder = TraceEncoder(
        span_dim=batch['span_features'].shape[-1],
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_gat_layers=2,
    ).to(device)

    span_features = batch['span_features'].to(device)
    adjacency = batch['trace_adjacency'].to(device)

    trace_output = trace_encoder({
    "span_features": span_features,
    "trace_adjacency": adjacency,
    })
    print(f"  English text removed.: span_features {span_features.shape}")
    print(f"  English text removed.: pooled {trace_output.pooled.shape}")
    print(f"  English text removed.: {trace_output.layer_assignments.shape}")

                                    
    print("\n=== Log Encoder (DistilBERT) ===")
    log_encoder = LogEncoder(
        output_dim=256,
        pretrained_model="distilbert-base-uncased",
        freeze_bert=True,              
    ).to(device)

    log_tokens = batch['log_tokens'].to(device)
    log_mask = batch['log_mask'].to(device)

    log_output = log_encoder({
    "log_tokens": log_tokens,
    "log_mask": log_mask,
    })
    print(f"  English text removed.: log_tokens {log_tokens.shape}")
    print(f"  English text removed.: pooled {log_output.pooled.shape}")

                                     
    print("\n=== Gaussian Positional Encoding ===")
    gaussian = GaussianPositionalEncoding(initial_sigma=30.0).to(device)

    rel_positions = batch['relative_time_positions'].to(device)
    attention_weights = gaussian(rel_positions)
    print(f"  English text removed.: relative_positions {rel_positions.shape}")
    print(f"  English text removed.: attention_weights {attention_weights.shape}")
    print(f"  English text removed.: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")

    print('English text removed.')

if __name__ == "__main__":
    main()
