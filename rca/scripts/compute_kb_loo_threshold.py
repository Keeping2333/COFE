#!/usr/bin/env python3
"""KB Leave-One-Out calibration for the B1 retrieval-trust threshold τ.

The B1 override rule in pipeline.py uses top-1 cosine similarity as a
retrieval-confidence score: when sim ≥ τ, the top-K plurality fault_type
overrides the LLM's fault_type prediction.

τ is calibrated on TRAINING data only via leave-one-out:
  - For each of the 288 KB vectors v_i:
      pop v_i, take top-1 cosine sim against remaining same-layer vectors
  - Take the chosen percentile (default p1) of the 288-sample LOO distribution

The same-layer LOO matches the inference behavior, where retrieval is
restricted by the Layer Classifier's prediction. Global (cross-layer) LOO
is also computed for comparison; it tends to be higher because the layer
filter already excludes near neighbors in other layers.

No test-set statistic is used.

Usage:
    cd <artifact-root>
    conda activate fullstack_rca
    python rca/scripts/compute_kb_loo_threshold.py \\
        --kb-dir knowledge_base_6span_agg \\
        --ckpt-dir checkpoints_6span \\
        --percentile 1.0
"""
from __future__ import annotations

import argparse
import json
import os

import faiss
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="KB-LOO τ calibration for B1 retrieval-trust gate")
    p.add_argument("--kb-dir", default="knowledge_base_6span_agg",
                   help="KB directory containing index.faiss + metadata.json")
    p.add_argument("--ckpt-dir", default="checkpoints_6span",
                   help="Where retrieval_tau.json will be written")
    p.add_argument("--percentile", type=float, default=1.0,
                   help="Percentile of the layer-filtered LOO distribution used as τ")
    p.add_argument("--output", default=None,
                   help="Explicit path for the JSON output "
                        "(default: <ckpt-dir>/retrieval_tau.json)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"KB:   {args.kb_dir}")
    print(f"Ckpt: {args.ckpt_dir}")
    print(f"τ percentile: p{args.percentile}\n")

                                                                        
                                
                                                                        
    idx = faiss.read_index(os.path.join(args.kb_dir, "index.faiss"))
    n = int(idx.ntotal)
    vecs = idx.reconstruct_n(0, n).astype(np.float32)
                                                                          
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    with open(os.path.join(args.kb_dir, "metadata.json")) as f:
        meta = json.load(f)
    id_list = meta["id_list"]
    meta_map = meta["metadata"]
    if len(id_list) != n:
        raise RuntimeError(f"id_list len {len(id_list)} != FAISS ntotal {n}")
    layers = np.array([meta_map[i].get("layer", "?") for i in id_list])
    fts = np.array([meta_map[i].get("fault_type", "?") for i in id_list])

                                                                        
                                                             
                                                                        
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -np.inf)

                                                                        
                                                     
                                                                        
    layer_filtered_top1 = np.full(n, -np.inf, dtype=np.float32)
    lf_neighbor_ft = np.empty(n, dtype=object)
    for i in range(n):
        mask = (layers == layers[i])
        mask[i] = False                
        row = np.where(mask, sim[i], -np.inf)
        j = int(row.argmax())
        layer_filtered_top1[i] = row[j]
        lf_neighbor_ft[i] = fts[j]
    lf_same_ft = (lf_neighbor_ft == fts)

                                                                        
                                                        
                                                                        
    global_top1 = sim.max(axis=1)
    global_neighbor_idx = sim.argmax(axis=1)
    global_neighbor_layer = layers[global_neighbor_idx]
    global_neighbor_ft = fts[global_neighbor_idx]
    global_same_ft = (global_neighbor_ft == fts)
    global_same_layer = (global_neighbor_layer == layers)

                                                                        
                                                     
                                                                        
    ps = [0.5, 1, 2, 5, 10, 25, 50, 75, 95, 99]

    def pct_dict(arr, ps=ps):
        return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

    def summary(arr):
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            **pct_dict(arr),
        }

    lf_summary = summary(layer_filtered_top1)
    gl_summary = summary(global_top1)

    by_layer = {}
    for L in sorted(set(layers.tolist())):
        m = (layers == L)
        if not m.any():
            continue
        v = layer_filtered_top1[m]
        by_layer[L] = {
            "n": int(m.sum()),
            "min": float(v.min()),
            "p1": float(np.percentile(v, 1)),
            "p5": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)),
            "max": float(v.max()),
            "same_ft_rate": float(lf_same_ft[m].mean()),
        }

    by_ft = {}
    for ft in sorted(set(fts.tolist())):
        m = (fts == ft)
        v = layer_filtered_top1[m]
        by_ft[ft] = {
            "n": int(m.sum()),
            "min": float(v.min()),
            "p1": float(np.percentile(v, 1)),
            "p5": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)),
            "max": float(v.max()),
            "same_ft_rate": float(lf_same_ft[m].mean()),
        }

    tau = float(np.percentile(layer_filtered_top1, args.percentile))

                                                                        
                   
                                                                        
    print(f"KB size: {n} vectors  ({len(by_layer)} layers × {len(by_ft)} FTs)\n")
    print("=" * 72)
    print("Layer-filtered LOO top-1 sim (matches inference layer filter)")
    print("=" * 72)
    print(f"  n={lf_summary['n']}  mean={lf_summary['mean']:.4f}  std={lf_summary['std']:.4f}")
    print(f"  min={lf_summary['min']:.4f}  max={lf_summary['max']:.4f}")
    print(f"  Percentiles:")
    for p in ps:
        print(f"    p{p:<4} = {lf_summary[f'p{p}']:.4f}")
    print(f"  Same-FT 1-NN rate (LOO): {lf_same_ft.mean()*100:.1f}%")
    print()
    print(f"  By layer:")
    for L, s in by_layer.items():
        print(f"    {L}: n={s['n']:3d}  min={s['min']:.4f}  p1={s['p1']:.4f}  p5={s['p5']:.4f}  "
              f"p50={s['p50']:.4f}  same_ft={s['same_ft_rate']*100:5.1f}%")
    print()
    print(f"  By fault_type:")
    for ft, s in by_ft.items():
        print(f"    {ft:32s} n={s['n']:2d}  min={s['min']:.4f}  p1={s['p1']:.4f}  "
              f"p5={s['p5']:.4f}  same_ft={s['same_ft_rate']*100:5.1f}%")
    print()
    print("=" * 72)
    print("Global LOO top-1 sim (no layer filter — diagnostic only)")
    print("=" * 72)
    print(f"  n={gl_summary['n']}  mean={gl_summary['mean']:.4f}  std={gl_summary['std']:.4f}")
    print(f"  min={gl_summary['min']:.4f}  max={gl_summary['max']:.4f}")
    for p in ps:
        print(f"    p{p:<4} = {gl_summary[f'p{p}']:.4f}")
    print(f"  Same-layer 1-NN rate: {global_same_layer.mean()*100:.1f}%")
    print(f"  Same-FT    1-NN rate: {global_same_ft.mean()*100:.1f}%")
    print()
    print("=" * 72)
    print(f"τ = layer-filtered LOO p{args.percentile} = {tau:.4f}")
    print("=" * 72)

                                                                        
                
                                                                        
    result = {
        "kb_dir": args.kb_dir,
        "kb_size": n,
        "calibration_percentile": args.percentile,
        "tau": tau,
        "tau_derivation": (
            f"p{args.percentile} percentile of layer-filtered KB leave-one-out "
            f"top-1 cosine similarity distribution (n={n}). Matches inference "
            "behavior where retrieval is restricted to the Layer Classifier's "
            "predicted layer. No test-set statistic involved."
        ),
        "layer_filtered_loo": {
            **lf_summary,
            "same_ft_neighbor_rate": float(lf_same_ft.mean()),
            "by_layer": by_layer,
            "by_fault_type": by_ft,
        },
        "global_loo": {
            **gl_summary,
            "same_layer_neighbor_rate": float(global_same_layer.mean()),
            "same_ft_neighbor_rate": float(global_same_ft.mean()),
            "note": "Diagnostic only; B1 uses layer_filtered_loo for τ.",
        },
    }
    out_path = args.output or os.path.join(args.ckpt_dir, "retrieval_tau.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
