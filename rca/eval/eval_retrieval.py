#!/usr/bin/env python3
'English text removed.'
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.fusion.projector import MultiModalProjector
from fullstack_rca.rag.vectorstore import FaissVectorStore

EMBEDDING_DIM = 512
K_VALUES = [1, 3, 5, 10, 20]


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval quality evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Source domain config (checkpoint + KB). Default: rca/config/dataset.yaml")
    parser.add_argument("--eval-mode", type=str, default="in_domain",
                        choices=["in_domain", "cross_domain"],
                        help="Evaluation mode (default: in_domain)")
    parser.add_argument("--query-config", type=str, default=None,
                        help="Target domain dataset config (cross_domain mode only)")
    parser.add_argument("--label-mapping", type=str, default=None,
                        help="Path to label_mapping.yaml (cross_domain mode)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Max query samples for cross_domain (0 = all)")
    return parser.parse_args()


# ===== Label mapping =====

def load_label_mapping(mapping_path, dataset_key="tt_data"):
    if not mapping_path or not os.path.exists(mapping_path):
        return {}
    with open(mapping_path) as f:
        raw = yaml.safe_load(f)
    return raw.get(dataset_key, {})


def map_label(gt_ft, mapping):
    """Map a query-domain fault_type to source-domain canonical labels."""
    if not mapping:
        return None, None, "original"
    entry = mapping.get(gt_ft)
    if entry:
        return entry["layer"], entry["fault_type"], entry.get("confidence", "coarse")
    return None, None, "unmapped"


# ===== Model loading =====

def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct trace encoder rich-feature config from checkpoint (T1+T2).
    trace_cfg = checkpoint.get("trace_encoder_config", {})
    trace_encoder = TraceEncoder(
        span_dim=int(trace_cfg.get("span_dim", 6)),
        hidden_dim=128, output_dim=256,
        num_heads=4, num_gat_layers=2,
        enable_rich_features=bool(trace_cfg.get("enable_rich_features", False)),
        num_service_buckets=int(trace_cfg.get("num_service_buckets", 64)),
        num_operation_buckets=int(trace_cfg.get("num_operation_buckets", 1024)),
        service_embed_dim=int(trace_cfg.get("service_embed_dim", 16)),
        operation_embed_dim=int(trace_cfg.get("operation_embed_dim", 32)),
        status_embed_dim=int(trace_cfg.get("status_embed_dim", 4)),
        num_status_classes=int(trace_cfg.get("num_status_classes", 4)),
        num_error_type_buckets=int(trace_cfg.get("num_error_type_buckets", 32)),
        error_type_embed_dim=int(trace_cfg.get("error_type_embed_dim", 8)),
        num_otel_status_classes=int(trace_cfg.get("num_otel_status_classes", 3)),
        otel_status_embed_dim=int(trace_cfg.get("otel_status_embed_dim", 4)),
    ).to(device)
    log_encoder = LogEncoder(
        output_dim=256, pretrained_model="distilbert-base-uncased", freeze_bert=True,
    ).to(device)
    ckpt_num_metrics = checkpoint.get("num_metrics")
    if ckpt_num_metrics is None:
        raise KeyError(
            "Checkpoint missing 'num_metrics'. Re-train with the round2 "
            "train.py to produce a checkpoint that records this field."
        )
    metric_encoder = create_metric_encoder(
        num_metrics=int(ckpt_num_metrics), output_dim=256, initial_sigma=30.0,
    ).to(device)
    projector = MultiModalProjector(
        log_dim=256, trace_dim=256, metric_dim=256,
        output_dim=EMBEDDING_DIM, fusion_method="concat",
    ).to(device)

    trace_encoder.load_state_dict(checkpoint["trace_encoder"])
    log_encoder.load_state_dict(checkpoint["log_encoder"])
    metric_encoder.load_state_dict(checkpoint["metric_encoder"])
    projector.load_state_dict(checkpoint["projector"])

    trace_encoder.eval()
    log_encoder.eval()
    metric_encoder.eval()
    projector.eval()

    return trace_encoder, log_encoder, metric_encoder, projector


@torch.no_grad()
def encode_sample(sample, trace_enc, log_enc, metric_enc, projector, device):
    """Encode a single sample to a fused embedding [512]."""
    trace_inputs = {
        "span_features": sample.span_features.unsqueeze(0).to(device),
        "trace_adjacency": sample.trace_adjacency.unsqueeze(0).to(device),
    }
    for attr in (
        "span_service_ids", "span_operation_ids",
        "span_status_classes", "span_is_root", "span_is_leaf",
        "span_is_on_critical_path",
    ):
        v = getattr(sample, attr, None)
        if v is not None:
            trace_inputs[attr] = v.unsqueeze(0).to(device)
    trace_emb = trace_enc(trace_inputs).embeddings
    log_emb = log_enc({
        "log_tokens": sample.log_tokens.unsqueeze(0).to(device),
        "log_mask": sample.log_attention_mask.unsqueeze(0).to(device),
    }).embeddings
    metric_emb, _ = metric_enc(
        metric_values=sample.metric_values.unsqueeze(0).to(device),
        relative_time_positions=sample.relative_time_positions.unsqueeze(0).to(device),
    )
    fused = projector(log_emb, metric_emb, trace_emb)
    return fused["fused"].cpu().numpy().squeeze(0)


# ===== In-domain evaluation (leave-one-out) =====

def load_kb(kb_dir):
    store = FaissVectorStore.load(kb_dir)
    print(f"Loaded KB: {store.count()} vectors, dim={store.embedding_dim}")
    return store


def evaluate_in_domain(store, k_values):
    n = store.count()
    max_k = max(k_values)

    print(f"\nRunning leave-one-out evaluation on {n} samples...")
    print(f"K values: {k_values}")

    all_vectors = np.zeros((n, store.embedding_dim), dtype=np.float32)
    for i in range(n):
        all_vectors[i] = store._index.reconstruct(i)

    start = time.time()
    distances, indices = store._index.search(all_vectors, max_k + 1)
    print(f"Batch search time: {time.time() - start:.3f}s ({n} queries)")

    id_list = store._id_list
    metadata = store._metadata

    layer_groups = defaultdict(set)
    ft_groups = defaultdict(set)
    for i, doc_id in enumerate(id_list):
        meta = metadata.get(doc_id, {})
        layer_groups[meta.get("layer", "")].add(i)
        ft_groups[meta.get("fault_type", "")].add(i)

    print(f"\nLayer distribution:")
    for layer in sorted(layer_groups.keys()):
        print(f"  {layer}: {len(layer_groups[layer])}")
    print(f"Fault type distribution:")
    for ft in sorted(ft_groups.keys()):
        print(f"  {ft}: {len(ft_groups[ft])}")

    results = {}
    for k in k_values:
        layer_hits = ft_hits = 0
        layer_recall_sum = ft_recall_sum = 0.0
        layer_mrr_sum = ft_mrr_sum = 0.0

        for i in range(n):
            query_meta = metadata.get(id_list[i], {})
            query_layer = query_meta.get("layer", "")
            query_ft = query_meta.get("fault_type", "")

            topk_indices = []
            for j in range(max_k + 1):
                idx = int(indices[i, j])
                if idx != i and idx >= 0:
                    topk_indices.append(idx)
                if len(topk_indices) >= k:
                    break

            same_layer = same_ft = 0
            first_layer_rank = first_ft_rank = None

            for rank, idx in enumerate(topk_indices):
                ret_meta = metadata.get(id_list[idx], {})
                if ret_meta.get("layer", "") == query_layer:
                    same_layer += 1
                    if first_layer_rank is None:
                        first_layer_rank = rank + 1
                if ret_meta.get("fault_type", "") == query_ft:
                    same_ft += 1
                    if first_ft_rank is None:
                        first_ft_rank = rank + 1

            if same_layer > 0:
                layer_hits += 1
            if same_ft > 0:
                ft_hits += 1

            total_same_layer = len(layer_groups[query_layer]) - 1
            total_same_ft = len(ft_groups[query_ft]) - 1
            if total_same_layer > 0:
                layer_recall_sum += same_layer / min(k, total_same_layer)
            if total_same_ft > 0:
                ft_recall_sum += same_ft / min(k, total_same_ft)

            if first_layer_rank is not None:
                layer_mrr_sum += 1.0 / first_layer_rank
            if first_ft_rank is not None:
                ft_mrr_sum += 1.0 / first_ft_rank

        results[k] = {
            "hit_at_k_layer": layer_hits / n,
            "hit_at_k_ft": ft_hits / n,
            "recall_at_k_layer": layer_recall_sum / n,
            "recall_at_k_ft": ft_recall_sum / n,
            "mrr_layer": layer_mrr_sum / n,
            "mrr_ft": ft_mrr_sum / n,
        }

    return results


def evaluate_per_layer(store):
    n = store.count()
    id_list = store._id_list
    metadata = store._metadata

    all_vectors = np.zeros((n, store.embedding_dim), dtype=np.float32)
    for i in range(n):
        all_vectors[i] = store._index.reconstruct(i)

    distances, indices = store._index.search(all_vectors, 11)

    layer_indices = defaultdict(list)
    for i, doc_id in enumerate(id_list):
        layer = metadata.get(doc_id, {}).get("layer", "")
        layer_indices[layer].append(i)

    print("\n=== Per-Layer Hit@5 (by fault_type) ===")
    print(f"{'Layer':<10} {'Count':<8} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10}")
    print("-" * 50)

    for layer in sorted(layer_indices.keys()):
        sample_indices = layer_indices[layer]
        count = len(sample_indices)

        for k in [1, 3, 5]:
            hits = 0
            for i in sample_indices:
                query_ft = metadata.get(id_list[i], {}).get("fault_type", "")
                topk = []
                for j in range(11):
                    idx = int(indices[i, j])
                    if idx != i and idx >= 0:
                        topk.append(idx)
                    if len(topk) >= k:
                        break
                for idx in topk:
                    if metadata.get(id_list[idx], {}).get("fault_type", "") == query_ft:
                        hits += 1
                        break

            if k == 1:
                h1 = hits / count
            elif k == 3:
                h3 = hits / count
            else:
                h5 = hits / count

        print(f"{layer:<10} {count:<8} {h1:<10.4f} {h3:<10.4f} {h5:<10.4f}")


# ===== Cross-domain evaluation =====

def evaluate_cross_domain(store, query_dataset, models, device, label_mapping, k_values, max_samples=0):
    """Encode query dataset samples and search against source-domain KB."""
    trace_enc, log_enc, metric_enc, projector = models

    n_query = len(query_dataset)
    if max_samples > 0:
        n_query = min(n_query, max_samples)

    print(f"\nEncoding {n_query} query samples...")
    query_vectors = []
    query_labels = []

    for i in tqdm(range(n_query), desc="Encoding queries"):
        sample = query_dataset[i]
        emb = encode_sample(sample, trace_enc, log_enc, metric_enc, projector, device)
        query_vectors.append(emb)

        gt_layer = sample.layer if sample.layer else "Normal"
        gt_ft = sample.root_cause_type if sample.root_cause_type else "normal"

        mapped_layer, mapped_ft, conf = map_label(gt_ft, label_mapping)
        if conf == "unmapped" or conf == "original":
            mapped_layer = gt_layer
            mapped_ft = gt_ft

        query_labels.append({
            "layer_raw": gt_layer,
            "ft_raw": gt_ft,
            "layer": mapped_layer,
            "ft": mapped_ft,
            "confidence": conf,
        })

    query_matrix = np.stack(query_vectors).astype(np.float32)

    # Search KB
    max_k = max(k_values)
    print(f"Searching KB ({store.count()} entries) with {n_query} queries, top-{max_k}...")
    start = time.time()
    distances, indices = store._index.search(query_matrix, max_k)
    print(f"Search time: {time.time() - start:.3f}s")

    kb_id_list = store._id_list
    kb_metadata = store._metadata

    # Print query label distribution
    from collections import Counter as _Counter
    layer_dist = _Counter(q["layer"] for q in query_labels)
    ft_dist = _Counter(q["ft"] for q in query_labels)
    conf_dist = _Counter(q["confidence"] for q in query_labels)

    print(f"\nQuery distribution (mapped):")
    print(f"  Layers: {dict(sorted(layer_dist.items()))}")
    print(f"  Mapping confidence: {dict(sorted(conf_dist.items()))}")

    # KB label groups for recall computation
    kb_layer_counts = defaultdict(int)
    kb_ft_counts = defaultdict(int)
    for doc_id in kb_id_list:
        meta = kb_metadata.get(doc_id, {})
        kb_layer_counts[meta.get("layer", "")] += 1
        kb_ft_counts[meta.get("fault_type", "")] += 1

    print(f"\nKB distribution:")
    for layer in sorted(kb_layer_counts.keys()):
        print(f"  {layer}: {kb_layer_counts[layer]}")

    # Compute metrics
    results = {}
    for k in k_values:
        layer_hits = ft_hits = 0
        layer_mrr_sum = ft_mrr_sum = 0.0
        layer_recall_sum = ft_recall_sum = 0.0
        evaluated = 0

        for qi in range(n_query):
            ql = query_labels[qi]
            q_layer = ql["layer"]
            q_ft = ql["ft"]

            topk_indices = []
            for j in range(max_k):
                idx = int(indices[qi, j])
                if idx >= 0:
                    topk_indices.append(idx)
                if len(topk_indices) >= k:
                    break

            same_layer = same_ft = 0
            first_layer_rank = first_ft_rank = None

            for rank, idx in enumerate(topk_indices):
                ret_meta = kb_metadata.get(kb_id_list[idx], {})
                if ret_meta.get("layer", "") == q_layer:
                    same_layer += 1
                    if first_layer_rank is None:
                        first_layer_rank = rank + 1
                if ret_meta.get("fault_type", "") == q_ft:
                    same_ft += 1
                    if first_ft_rank is None:
                        first_ft_rank = rank + 1

            if same_layer > 0:
                layer_hits += 1
            if same_ft > 0:
                ft_hits += 1

            total_kb_layer = kb_layer_counts.get(q_layer, 0)
            total_kb_ft = kb_ft_counts.get(q_ft, 0)
            if total_kb_layer > 0:
                layer_recall_sum += same_layer / min(k, total_kb_layer)
            if total_kb_ft > 0:
                ft_recall_sum += same_ft / min(k, total_kb_ft)

            if first_layer_rank is not None:
                layer_mrr_sum += 1.0 / first_layer_rank
            if first_ft_rank is not None:
                ft_mrr_sum += 1.0 / first_ft_rank

            evaluated += 1

        results[k] = {
            "hit_at_k_layer": layer_hits / evaluated,
            "hit_at_k_ft": ft_hits / evaluated,
            "recall_at_k_layer": layer_recall_sum / evaluated,
            "recall_at_k_ft": ft_recall_sum / evaluated,
            "mrr_layer": layer_mrr_sum / evaluated,
            "mrr_ft": ft_mrr_sum / evaluated,
            "n_evaluated": evaluated,
        }

    return results, query_labels


def print_results_table(results, k_values, title="Retrieval Metrics"):
    print(f"\n=== {title} ===")
    print(f"{'K':<6} {'Hit@K(layer)':<15} {'Hit@K(ft)':<15} {'Recall@K(layer)':<18} {'Recall@K(ft)':<18} {'MRR(layer)':<12} {'MRR(ft)':<12}")
    print("-" * 100)
    for k in k_values:
        r = results[k]
        print(
            f"{k:<6} "
            f"{r['hit_at_k_layer']:<15.4f} "
            f"{r['hit_at_k_ft']:<15.4f} "
            f"{r['recall_at_k_layer']:<18.4f} "
            f"{r['recall_at_k_ft']:<18.4f} "
            f"{r['mrr_layer']:<12.4f} "
            f"{r['mrr_ft']:<12.4f}"
        )


# ===== Main =====

def main():
    args = parse_args()
    start = time.time()

    source_config = load_config(args.config)
    kb_dir = source_config.get("kb_dir", "knowledge_base")

    print("=== Retrieval Evaluation ===\n")
    print(f"Mode: {args.eval_mode}")

    store = load_kb(kb_dir)

    if args.eval_mode == "in_domain":
        results = evaluate_in_domain(store, K_VALUES)
        print_results_table(results, K_VALUES, "Global Retrieval Metrics (In-Domain)")
        evaluate_per_layer(store)

    elif args.eval_mode == "cross_domain":
        if not args.query_config:
            print("ERROR: --query-config is required for cross_domain mode.")
            sys.exit(1)

        query_config = load_config(args.query_config)
        print(f"Query dataset: {query_config['dataset']['type']} ({query_config['dataset']['path']})")

        checkpoint_dir = source_config.get("checkpoint_dir", "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        print("\nLoading models...")
        models = load_models(checkpoint_path, device)

        print("Loading query dataset...")
        query_dataset = create_dataset(query_config["dataset"])
        print(f"  Query samples: {len(query_dataset)}")

        label_mapping = load_label_mapping(args.label_mapping)

        results, query_labels = evaluate_cross_domain(
            store, query_dataset, models, device, label_mapping, K_VALUES,
            max_samples=args.num_samples,
        )
        print_results_table(results, K_VALUES, "Cross-Domain Retrieval Metrics")

        # Per-confidence-level breakdown
        from collections import Counter as _Counter
        conf_groups = defaultdict(list)
        for i, ql in enumerate(query_labels):
            conf_groups[ql["confidence"]].append(i)

        if len(conf_groups) > 1:
            print("\n=== Accuracy by Mapping Confidence ===")
            for conf_type in ["exact", "coarse", "unmapped", "original"]:
                if conf_type not in conf_groups:
                    continue
                idxs = conf_groups[conf_type]
                print(f"\n  [{conf_type}] ({len(idxs)} samples)")

    elapsed = time.time() - start
    print(f"\nEvaluation completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
