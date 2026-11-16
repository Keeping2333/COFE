#!/usr/bin/env python3
'English text removed.'
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

        
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fullstack_rca.data.factory import create_dataset, load_config
from fullstack_rca.models.encoders.gaussian_attention import create_metric_encoder
from fullstack_rca.models.encoders.log_encoder import LogEncoder
from fullstack_rca.models.encoders.trace_encoder import TraceEncoder
from fullstack_rca.models.fusion.projector import MultiModalProjector
from fullstack_rca.rag.vectorstore import FaissVectorStore, Document
from fullstack_rca.rag.knowledge.episode import Episode
from fullstack_rca.rag.knowledge.manager import KnowledgeManager
from fullstack_rca.data.split import experiment_level_split, get_experiment_indices

      
EMBEDDING_DIM = 512                                      


def load_models(checkpoint_path: str, device: torch.device, *, span_dim: int, num_metrics: int):
    'English text removed.'
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Please run scripts/train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    duration_stats = checkpoint.get("duration_stats")
    if duration_stats:
        print(f"  Duration stats: mean={duration_stats['mean']:.4f}, std={duration_stats['std']:.4f}")

                                                                            
                                                                                      
    trace_cfg = checkpoint.get("trace_encoder_config", {})
    trace_encoder = TraceEncoder(
        span_dim=trace_cfg.get("span_dim", span_dim),
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
        output_dim=256,
        pretrained_model="distilbert-base-uncased",
        freeze_bert=True,
        unfreeze_last_n=0,                          
    ).to(device)

    metric_encoder = create_metric_encoder(
        num_metrics=num_metrics, output_dim=256, initial_sigma=30.0,
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

    print("  Models loaded and set to eval mode.")
    return trace_encoder, log_encoder, metric_encoder, projector, duration_stats


@torch.no_grad()
def encode_sample_single(
    sample: "MultimodalSample",
    trace_encoder, log_encoder, metric_encoder, projector,
    device: torch.device,
) -> tuple:
    'English text removed.'
    trace_inputs = {
        "span_features": sample.span_features.unsqueeze(0).to(device),
        "trace_adjacency": sample.trace_adjacency.unsqueeze(0).to(device),
    }
    for attr, key in (
        ("span_service_ids", "span_service_ids"),
        ("span_operation_ids", "span_operation_ids"),
        ("span_status_classes", "span_status_classes"),
        ("span_is_root", "span_is_root"),
        ("span_is_leaf", "span_is_leaf"),
        ("span_is_on_critical_path", "span_is_on_critical_path"),
    ):
        v = getattr(sample, attr, None)
        if v is not None:
            trace_inputs[key] = v.unsqueeze(0).to(device)
    trace_emb = trace_encoder(trace_inputs).embeddings

    log_emb = log_encoder({
        "log_tokens": sample.log_tokens.unsqueeze(0).to(device),
        "log_mask": sample.log_attention_mask.unsqueeze(0).to(device),
    }).embeddings

    metric_emb, _ = metric_encoder(
        metric_values=sample.metric_values.unsqueeze(0).to(device),
        relative_time_positions=sample.relative_time_positions.unsqueeze(0).to(device),
    )

    fused = projector(log_emb, metric_emb, trace_emb)
    return (
        fused["fused"].cpu().numpy().squeeze(0),                               
        trace_emb.cpu().numpy().squeeze(0),                                    
        log_emb.cpu().numpy().squeeze(0),                                    
    )


@torch.no_grad()
def encode_dataset(
    dataset,
    trace_encoder, log_encoder, metric_encoder, projector,
    device: torch.device,
    indices: list = None,
) -> tuple:
    'English text removed.'
    all_embeddings = []
    all_trace_embeddings = []
    all_log_embeddings = []
    all_infos = []
    samples_index = dataset._samples_index
    if indices is None:
        indices = list(range(len(dataset)))

    for i in tqdm(indices, desc="Encoding"):
        sample = dataset[i]
        info = samples_index[i]

        emb, trace_emb, log_emb = encode_sample_single(
            sample, trace_encoder, log_encoder, metric_encoder, projector, device,
        )
        all_embeddings.append(emb)
        all_trace_embeddings.append(trace_emb)
        all_log_embeddings.append(log_emb)

                                                    
        experiment_name = info.get("run_id", info.get("scenario_name", "unknown"))

                                                            
                                                                                                            
        sf = sample.span_features.numpy()
        error_rate = sf[:, 1].mean()                                   
        max_latency = sf[:, 0].max()                                       
        anomaly_score = error_rate + max_latency                           

        all_infos.append({
            "trace_id": sample.trace_id,
            "layer": sample.layer,
            "fault_layer": sample.fault_layer,
            "root_cause_type": info["root_cause"],
            "experiment_name": experiment_name,
            "anomaly_score": float(anomaly_score),
        })

    embeddings = np.stack(all_embeddings, axis=0)
    trace_embeddings = np.stack(all_trace_embeddings, axis=0)
    log_embeddings = np.stack(all_log_embeddings, axis=0)
    print(f"  Encoded {embeddings.shape[0]} samples, fused: {embeddings.shape}, "
          f"trace: {trace_embeddings.shape}, log: {log_embeddings.shape}")
    return embeddings, trace_embeddings, log_embeddings, all_infos


def aggregate_by_experiment(
    embeddings: np.ndarray,
    sample_infos: list,
    trace_embeddings: np.ndarray,
    log_embeddings: np.ndarray,
) -> tuple:
    """
    Aggregate trace-level embeddings to experiment-level via anomaly-weighted pooling.

    Traces with higher anomaly scores (more errors, higher latency) get higher weight,
    reducing dilution of fault-specific signals by normal traces. The trace-only and
    log-only embeddings are pooled with the SAME per-experiment weights as the fused
    vectors, so all views stay aligned per experiment.

    Returns:
        agg_embeddings: np.ndarray [num_experiments, EMBEDDING_DIM]
        agg_trace:      np.ndarray [num_experiments, TRACE_DIM]
        agg_log:        np.ndarray [num_experiments, LOG_DIM]
        agg_infos: list[dict] - one entry per experiment
    """
    from collections import defaultdict
    exp_groups = defaultdict(list)

    for i, info in enumerate(sample_infos):
        exp_name = info["experiment_name"]
        exp_groups[exp_name].append(i)

    agg_embeddings = []
    agg_trace = []
    agg_log = []
    agg_infos = []

    for exp_name, indices in sorted(exp_groups.items()):
        exp_embs = embeddings[indices]
        exp_trace = trace_embeddings[indices]
        exp_log = log_embeddings[indices]

                                      
        scores = np.array([
            sample_infos[i].get("anomaly_score", 0.0) for i in indices
        ])

                                                                  
                                                                    
        floor = 0.01
        weights = np.maximum(scores, floor)
        weights = weights / weights.sum()                      

        def _pool(arr):
            v = (arr * weights[:, np.newaxis]).sum(axis=0)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

                                                   
        rep_info = sample_infos[indices[0]].copy()
        rep_info["num_traces"] = len(indices)
        rep_info["trace_id"] = exp_name                        
        rep_info["anomaly_weight_stats"] = {
            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "nonzero_ratio": float((scores > floor).mean()),
        }

        agg_embeddings.append(_pool(exp_embs))
        agg_trace.append(_pool(exp_trace))
        agg_log.append(_pool(exp_log))
        agg_infos.append(rep_info)

                             
    all_max = [info["anomaly_weight_stats"]["max_score"] for info in agg_infos]
    all_nz = [info["anomaly_weight_stats"]["nonzero_ratio"] for info in agg_infos]
    print(f"  Aggregated {len(sample_infos)} traces -> {len(agg_infos)} experiments")
    print(f"  Anomaly score stats: max={np.max(all_max):.3f}, "
          f"avg_max={np.mean(all_max):.3f}, avg_nonzero_ratio={np.mean(all_nz):.3f}")

    return np.stack(agg_embeddings), np.stack(agg_trace), np.stack(agg_log), agg_infos


def build_knowledge_base(
    embeddings: np.ndarray,
    sample_infos: list,
    kb_dir: str,
    trace_embeddings: np.ndarray,
    log_embeddings: np.ndarray,
) -> KnowledgeManager:
    'English text removed.'
    os.makedirs(kb_dir, exist_ok=True)

                 
    vector_store = FaissVectorStore(embedding_dim=EMBEDDING_DIM)
    manager = KnowledgeManager(
        vector_store=vector_store,
        metadata_path=os.path.join(kb_dir, "episodes.json"),
    )

    print(f"Building KB with {len(sample_infos)} episodes...")

                   
    episodes = []
    episode_embeddings = []
    episode_trace_vecs = []
    episode_log_vecs = []
    episode_ids_order = []

    for i, info in enumerate(sample_infos):
        layer = info["layer"] if info["layer"] else "Normal"
        rca = info["root_cause_type"]
        experiment = info["experiment_name"]

                   
        if layer == "Normal":
            desc = f"Normal operation (experiment: {experiment})"
        else:
            desc = (
                f"{layer} fault: {rca} "
                f"(experiment: {experiment}, trace: {info['trace_id']})"
            )

        episode = Episode(
            id=f"ep_{i:05d}_{info['trace_id'][:16]}",
            experiment_id=experiment,
            layer=layer,
            fault_type=rca,
            description=desc,
            metadata={
                "experiment_name": experiment,
                "trace_id": info["trace_id"],
                "dataset_index": i,
            },
        )
        episodes.append(episode)
        episode_embeddings.append(embeddings[i])
        episode_trace_vecs.append(trace_embeddings[i])
        episode_log_vecs.append(log_embeddings[i])
        episode_ids_order.append(episode.id)

          
    episode_embeddings_np = np.stack(episode_embeddings)
    manager.add_episodes(episodes, episode_embeddings_np)

             
    vector_store.persist(kb_dir)

                                                             
                                                         
                                        
    def _save_vecs(vec_list, fname):
        arr = np.stack(vec_list).astype(np.float32)
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        np.save(os.path.join(kb_dir, fname), arr)
        return arr.shape

    ts = _save_vecs(episode_trace_vecs, "trace_vectors.npy")
    ls = _save_vecs(episode_log_vecs, "log_vectors.npy")
    with open(os.path.join(kb_dir, "ood_episode_ids.json"), "w") as f:
        json.dump(episode_ids_order, f)
    print(f"  OOD-fallback vectors saved: trace {ts}, log {ls} "
          f"(trace_vectors.npy + log_vectors.npy + ood_episode_ids.json)")

          
    stats = manager.get_stats()
    print(f"\n=== KB Statistics ===")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Vector store count: {stats['vector_store_count']}")
    print(f"  By layer:")
    for layer, count in sorted(stats["by_layer"].items()):
        print(f"    {layer}: {count}")
    print(f"  By fault type:")
    for ft, count in sorted(stats["by_fault_type"].items()):
        print(f"    {ft}: {count}")
    print(f"  KB saved to: {kb_dir}")

    return manager


def quick_sanity_check(manager: KnowledgeManager, embeddings: np.ndarray, sample_infos: list):
    'English text removed.'
    print("\n=== Quick Sanity Check ===")

                
    test_indices = []
    seen_layers = set()
    for i, info in enumerate(sample_infos):
        layer = info["layer"] if info["layer"] else "Normal"
        if layer not in seen_layers:
            seen_layers.add(layer)
            test_indices.append(i)
        if len(test_indices) >= 5:
            break

    for idx in test_indices:
        info = sample_infos[idx]
        query_emb = embeddings[idx]
        query_layer = info["layer"] if info["layer"] else "Normal"

        matches = manager.retrieve(query_emb, k=5)
        print(f"\nQuery: idx={idx}, layer={query_layer}, rca={info['root_cause_type']}")
        for j, m in enumerate(matches):
            print(
                f"  Top-{j+1}: {m.episode.id[:30]:30s} "
                f"layer={m.episode.layer:7s} "
                f"rca={m.episode.fault_type:25s} "
                f"sim={m.similarity:.4f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Build vector knowledge base")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dataset.yaml config (default: rca/config/dataset.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    kb_dir = config.get("kb_dir", "knowledge_base")

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {config['dataset']['type']} ({config['dataset']['path']})")

                                  
    print("\n[1/4] Loading dataset...")
    dataset = create_dataset(config["dataset"])
    print(f"  Dataset size: {len(dataset)} samples")

                            
    sample0 = dataset[0]
    span_dim = int(sample0.span_features.shape[-1])
    num_metrics = int(sample0.metric_values.shape[0])
    print(f"  Detected span_dim={span_dim}, num_metrics={num_metrics}")

             
    print("\n[2/4] Loading models...")
    trace_enc, log_enc, metric_enc, projector, duration_stats = load_models(
        checkpoint_path, device, span_dim=span_dim, num_metrics=num_metrics
    )

                                                       
    if duration_stats:
        dataset.duration_stats = duration_stats

                               
    print("\n[3/4] Encoding training experiments only...")
    train_exps, test_exps = experiment_level_split(dataset._samples_index, test_ratio=0.2, seed=42)
    train_indices = get_experiment_indices(dataset._samples_index, train_exps)
    print(f"  Training traces: {len(train_indices)} / {len(dataset)} total")
    embeddings, trace_embeddings, log_embeddings, sample_infos = encode_dataset(
        dataset, trace_enc, log_enc, metric_enc, projector, device,
        indices=train_indices,
    )

                                                                                
                                              
    print("\n[3.5/4] Aggregating to experiment level...")
    embeddings, trace_embeddings, log_embeddings, sample_infos = aggregate_by_experiment(
        embeddings, sample_infos, trace_embeddings, log_embeddings,
    )

              
    print("\n[4/4] Building knowledge base...")
    manager = build_knowledge_base(
        embeddings, sample_infos, kb_dir, trace_embeddings, log_embeddings,
    )

            
    quick_sanity_check(manager, embeddings, sample_infos)

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f}s")
    print(f"KB location: {kb_dir}")
    print(f"Files: index.faiss, metadata.json, episodes.json")


if __name__ == "__main__":
    main()
