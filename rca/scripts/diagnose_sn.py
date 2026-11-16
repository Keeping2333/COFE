#!/usr/bin/env python3
'English text removed.'
import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from typing import Dict, Optional

                                                                                    
                                                                                      
                                                                                
                                                                                 
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fullstack_rca.agent.llm_agent import LLMReasoningAgent
from fullstack_rca.data.sn_dataset import SNDataset
from fullstack_rca.rag.pipeline import RCAPipeline

                
SN_DATA_DIR    = os.environ.get("SN_DATA_DIR", "data/SN_data")
CHECKPOINT     = os.environ.get("RCA_CHECKPOINT", "checkpoints/best_model.pt")
KB_DIR         = os.environ.get("RCA_KB_DIR", "knowledge_base")


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot RCA on SN_data")
    p.add_argument("--per-scenario", type=int, default=5,
                   help='English text removed.')
    p.add_argument("--model",  default="qwen2.5:14b", help='English text removed.')
    p.add_argument("--top-k",  type=int, default=5,   help='English text removed.')
    p.add_argument("--verbose", action="store_true",  help='English text removed.')
    return p.parse_args()


def layer_match(pred_layer: str, gt_layer: Optional[str]) -> bool:
    'English text removed.'
    gt = gt_layer or "Normal"
    return (pred_layer or "").upper() == gt.upper()


def compute_metrics(results):
    'English text removed.'
    total = len(results)
    if total == 0:
        return {}

    layer_correct = sum(1 for r in results if r["layer_ok"])
    rca_correct   = sum(1 for r in results if r["rca_ok"])
    sop_pass      = sum(1 for r in results if r["sop_ok"])
    confidences   = [r["confidence"] for r in results]

                   
    per_layer: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt = r["gt_layer"] or "Normal"
        per_layer[gt]["total"] += 1
        if r["layer_ok"]:
            per_layer[gt]["correct"] += 1

                     
    confusion = Counter()
    for r in results:
        gt   = r["gt_layer"] or "Normal"
        pred = r["pred_layer"] or "Normal"
        confusion[(gt, pred)] += 1

    return {
        "total":          total,
        "layer_acc":      layer_correct / total,
        "rca_acc":        rca_correct / total,
        "mean_conf":      sum(confidences) / total,
        "sop_pass_rate":  sop_pass / total,
        "per_layer":      dict(per_layer),
        "confusion":      confusion,
    }


def print_metrics(metrics: Dict):
    print("\n" + "=" * 70)
    print("ZERO-SHOT EVALUATION RESULTS (SN_data → TT KB)")
    print("=" * 70)
    print(f"Total samples:         {metrics['total']}")
    print(f"Layer Accuracy:        {metrics['layer_acc']:.1%}  ({int(metrics['layer_acc']*metrics['total'])}/{metrics['total']})")
    print(f"Fault Type Accuracy:   {metrics['rca_acc']:.1%}  ({int(metrics['rca_acc']*metrics['total'])}/{metrics['total']})")
    print(f"Mean Confidence:       {metrics['mean_conf']:.3f}")
    print(f"SOP Validated:         {metrics['sop_pass_rate']:.1%}")

    print("\nPer-Layer Accuracy:")
    print(f"{'GT Layer':<12} {'Correct':<10} {'Total':<8} {'Accuracy'}")
    print("-" * 44)
    for layer, stat in sorted(metrics["per_layer"].items()):
        acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0
        print(f"{layer:<12} {stat['correct']:<10} {stat['total']:<8} {acc:.1%}")

    print("\nConfusion Matrix (GT → Pred):")
    for (gt, pred), cnt in sorted(metrics["confusion"].items()):
        tag = "[OK]" if gt == pred else "[MISS]"
        print(f"  {gt:>8} → {pred:<8} : {cnt:3d}  {tag}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Zero-Shot RCA: SN_data  (model trained on TT_data)")
    print("=" * 60)

                  
    print("\n[1/4] Loading SNDataset...")
    dataset = SNDataset(
        data_dir=SN_DATA_DIR,
        max_traces_per_scenario=args.per_scenario,
        max_spans_per_trace=30,
        metric_window_seconds=60.0,
    )
    stats = dataset.get_stats()
    print(f"  Total samples: {stats['total']}")
    print(f"  By layer: {stats['by_layer']}")
    print(f"  By RCA:   {stats['by_rca']}")

    if len(dataset) == 0:
        print("ERROR: no samples loaded, check SN_DATA_DIR")
        return

                     
    print("\n[2/4] Initializing RCAPipeline (TT_data checkpoint + KB)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = RCAPipeline(
        checkpoint_path=CHECKPOINT,
        kb_dir=KB_DIR,
        device=device,
        top_k=args.top_k,
    )
    print(f"  KB size: {pipeline.knowledge_manager.vector_store.count()} episodes")

                      
    print("\n[3/4] Initializing LLM Agent...")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    agent = LLMReasoningAgent(
        model_name=args.model,
        api_key=api_key if api_key else None,
    )
    print(f"  Model: {args.model}")

              
    print(f"\n[4/4] Diagnosing {len(dataset)} samples...")
    print("-" * 60)

    results = []
    errors  = 0
    t_start = time.time()

    for i in range(len(dataset)):
        info = dataset._samples_index[i]
        sample = dataset[i]

        gt_layer = info["layer"]
        gt_rca   = info["root_cause"]
        exp_id   = info["experiment_id"]

        if args.verbose:
            print(f"\n--- Sample {i+1}/{len(dataset)} ---")
            print(f"  Experiment: {exp_id}")
            print(f"  Trace ID:   {sample.trace_id[:40]}")
            print(f"  GT:         {gt_layer or 'Normal'} / {gt_rca}")

        t0 = time.time()
        try:
                                               
            context = pipeline.process_sample(sample, exclude_trace_id=None)

            if args.verbose:
                print(f"  Retrieved {len(context.retrieved_episodes)} episodes:")
                for ep in context.retrieved_episodes[:3]:
                    print(f"    {ep['layer']:6s} {ep['fault_type']:30s} sim={ep['similarity']:.4f}")

                    
            diagnosis = agent.diagnose(context)
            pred_layer = str(diagnosis.root_cause_layer).replace("FaultLayer.", "")
            pred_rca   = diagnosis.root_cause_type
            confidence = diagnosis.confidence
            sop_ok     = agent.validate_sop_compliance(diagnosis)

        except Exception as e:
            logger.error("Sample %d failed: %s", i, e)
            errors += 1
            pred_layer = "ERROR"
            pred_rca   = "error"
            confidence = 0.0
            sop_ok     = False

        elapsed = time.time() - t0
        layer_ok = (pred_layer or "Normal").upper() == (gt_layer or "Normal").upper()
        rca_ok   = (pred_rca  or "").lower() == (gt_rca or "").lower()

        results.append({
            "exp_id":      exp_id,
            "gt_layer":    gt_layer,
            "gt_rca":      gt_rca,
            "pred_layer":  pred_layer,
            "pred_rca":    pred_rca,
            "confidence":  confidence,
            "layer_ok":    layer_ok,
            "rca_ok":      rca_ok,
            "sop_ok":      sop_ok,
        })

        layer_tag = "[OK]" if layer_ok else "[MISS]"
        rca_tag   = "[OK]" if rca_ok   else "[MISS]"
        if args.verbose:
            print(f"  Pred: {pred_layer} / {pred_rca}")
            print(f"  Layer: {layer_tag}  RCA: {rca_tag}  "
                  f"Conf: {confidence:.2f}  Time: {elapsed:.1f}s")
        else:
            print(f"  [{i+1:3d}] GT={gt_layer or 'Normal':6s}/{gt_rca:25s}  "
                  f"Pred={pred_layer:6s}  {layer_tag}  ({elapsed:.0f}s)")

             
    total_time = time.time() - t_start
    metrics = compute_metrics(results)
    print_metrics(metrics)
    print(f"\nTotal time: {total_time:.1f}s  |  "
          f"Avg: {total_time/max(len(results),1):.1f}s/sample  |  "
          f"Errors: {errors}/{len(dataset)}")

              
    print("\nPer-Experiment Breakdown:")
    exp_results = defaultdict(list)
    for r in results:
        exp_results[r["exp_id"]].append(r)
    print(f"{'Experiment':<45} {'GT':>6} {'Pred':>6} {'LayerOK'}")
    print("-" * 70)
    for exp_id, rlist in sorted(exp_results.items()):
        gt   = rlist[0]["gt_layer"] or "Normal"
        pred = Counter(r["pred_layer"] for r in rlist).most_common(1)[0][0] or "Normal"
        ok   = sum(r["layer_ok"] for r in rlist)
        tag  = "✓" if ok == len(rlist) else f"{ok}/{len(rlist)}"
        print(f"  {exp_id[:43]:<43} {gt:>6} {pred:>6}  {tag}")


import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
