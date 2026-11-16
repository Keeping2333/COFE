#!/usr/bin/env python3
'English text removed.'
import argparse
from fullstack_rca.data.dataset import collate_multimodal_samples
from fullstack_rca.data.factory import create_dataset, load_config
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset.yaml config (default: rca/config/dataset.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ds_config = config["dataset"]

    print(f"English text removed.: type={ds_config['type']}, path={ds_config['path']}")

              
    dataset = create_dataset(ds_config)

    print(f"\nEnglish text removed.:")
    print(f"  - Traces English text removed.: {len(dataset)}")

    if hasattr(dataset, "get_stats"):
        stats = dataset.get_stats()
        print(f"  - English text removed.: {stats.get('by_layer', {})}")
        print(f"  - English text removed.: {stats.get('by_rca', {})}")

               
    sample = dataset[0]
    print(f"\nEnglish text removed.:")
    print(f"  - trace_id: {sample.trace_id}")
    print(f"  - span_features shape: {sample.span_features.shape}")
    print(f"  - trace_adjacency shape: {sample.trace_adjacency.shape}")
    print(f"  - num_logs: {sample.num_logs}")
    print(f"  - log_tokens shape: {sample.log_tokens.shape}")
    print(f"  - metric_values shape: {sample.metric_values.shape}")
    print(f"  - relative_time_positions shape: {sample.relative_time_positions.shape}")
    print(f"  - layer: {sample.layer}")
    print(f"  - fault_layer: {sample.fault_layer}")
    print(f"  - root_cause_type: {sample.root_cause_type}")

                      
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_multimodal_samples,
        num_workers=0,
    )

               
    batch = next(iter(loader))
    print(f"\nEnglish text removed.:")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  - {key}: {value.shape}")
        else:
            print(f"  - {key}: {type(value)} len={len(value)}")

    print('English text removed.')

if __name__ == "__main__":
    main()
