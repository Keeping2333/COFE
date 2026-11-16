'English text removed.'
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..agent.base import AgentContext
from ..data.dataset import MultimodalSample
from ..data.exp_runs_dataset import _METRIC_FILES
from ..models.encoders.gaussian_attention import create_metric_encoder
from ..models.encoders.log_encoder import LogEncoder
from ..models.encoders.trace_encoder import TraceEncoder
from ..models.fusion.projector import MultiModalProjector
from ..models.layer_classifier import LayerClassifier
from .knowledge.episode import EpisodeMatch
from .knowledge.manager import KnowledgeManager
from .vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RCAPipeline:
    'English text removed.'

    def __init__(
        self,
        checkpoint_path: str,
        kb_dir: str,
        device: Optional[torch.device] = None,
        top_k: int = 5,
        embedding_dim: int = 512,
        evidence_detector: Optional[Any] = None,
        metric_files: Optional[List[str]] = None,
    ) -> None:
        'English text removed.'
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.embedding_dim = embedding_dim
        # Optional baseline-relative evidence detector. When set (and a
        # detector_input is supplied to process_experiment), it replaces the
        # hard-rule telemetry summaries. None → identical pre-modification path.
        self.evidence_detector = evidence_detector
        # Per-system metric channel names. Default (None) -> TT _METRIC_FILES,
        # so existing callers are byte-identical. diagnose.py passes the active
        # dataset.metric_files for SN.
        self.METRIC_NAMES = [
            Path(f).stem for f in (metric_files or _METRIC_FILES)
        ]

              
        self._load_models(checkpoint_path)

                                 
        self._load_layer_classifier(checkpoint_path)

                                                      
        self._load_retrieval_tau(checkpoint_path)

               
        self._load_kb(kb_dir)

        logger.info(
            f"RCAPipeline initialized: device={self.device}, "
            f"KB={self.knowledge_manager.vector_store.count()} episodes, top_k={top_k}"
        )

    def _load_models(self, checkpoint_path: str) -> None:
        'English text removed.'
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.duration_stats = checkpoint.get("duration_stats")

        # Reconstruct trace encoder rich-feature config from checkpoint (T1+T2).
        # Missing key -> defaults match the pre-modification encoder.
        trace_cfg = checkpoint.get("trace_encoder_config", {})
        self._trace_enable_rich = bool(trace_cfg.get("enable_rich_features", False))
        self.trace_encoder = TraceEncoder(
            span_dim=int(trace_cfg.get("span_dim", 6)),
            hidden_dim=128, output_dim=256,
            num_heads=4, num_gat_layers=2,
            enable_rich_features=self._trace_enable_rich,
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
        ).to(self.device)

        self.log_encoder = LogEncoder(
            output_dim=256,
            pretrained_model="distilbert-base-uncased",
            freeze_bert=True,
            unfreeze_last_n=0,     # must match train.py
        ).to(self.device)

        # Reconstruct metric encoder with the channel count it was trained on.
        # Saved by train.py — required for round2 (60 channels). Fail loudly
        # if absent so we never silently load a mis-shaped checkpoint.
        ckpt_num_metrics = checkpoint.get("num_metrics")
        if ckpt_num_metrics is None:
            raise KeyError(
                "Checkpoint missing 'num_metrics'. Re-train with the round2 "
                "train.py to produce a checkpoint that records this field."
            )
        self.metric_encoder = create_metric_encoder(
            num_metrics=int(ckpt_num_metrics), output_dim=256, initial_sigma=30.0,
        ).to(self.device)

        self.projector = MultiModalProjector(
            log_dim=256, trace_dim=256, metric_dim=256,
            output_dim=self.embedding_dim, fusion_method="concat",
        ).to(self.device)

        self.trace_encoder.load_state_dict(checkpoint["trace_encoder"])
        self.log_encoder.load_state_dict(checkpoint["log_encoder"])
        self.metric_encoder.load_state_dict(checkpoint["metric_encoder"])
        self.projector.load_state_dict(checkpoint["projector"])

        self.trace_encoder.eval()
        self.log_encoder.eval()
        self.metric_encoder.eval()
        self.projector.eval()

        logger.info("Models loaded and set to eval mode.")

    def _load_layer_classifier(self, checkpoint_path: str) -> None:
        'English text removed.'
        classifier_path = os.path.join(
            os.path.dirname(checkpoint_path), "layer_classifier.pt"
        )
        if os.path.isfile(classifier_path):
            ckpt = torch.load(classifier_path, map_location=self.device, weights_only=False)
            cfg = ckpt["config"]
            self.layer_classifier = LayerClassifier(
                log_dim=cfg["log_dim"],
                metric_dim=cfg["metric_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
            ).to(self.device)
            self.layer_classifier.load_state_dict(ckpt["classifier"])
            self.layer_classifier.eval()
            acc = ckpt.get("best_test_acc")
            acc_str = f"{acc:.1%}" if acc is not None else "?"
            logger.info(f"Layer classifier loaded (test_acc={acc_str})")
        else:
            self.layer_classifier = None
            logger.info("No layer_classifier.pt found — layer prediction from retrieval only.")

    def _load_retrieval_tau(self, checkpoint_path: str) -> None:
        """Load the B1 retrieval-trust threshold from training-side KB-LOO calibration.

        Reads <ckpt_dir>/retrieval_tau.json (produced by
        compute_kb_loo_threshold.py) and stores `self.retrieval_tau`. When the
        file is absent, the B1 gate stays disabled and the pipeline behaves as
        if no override exists.
        """
        tau_path = os.path.join(
            os.path.dirname(checkpoint_path), "retrieval_tau.json"
        )
        if os.path.isfile(tau_path):
            with open(tau_path) as f:
                tau_data = json.load(f)
            self.retrieval_tau = float(tau_data["tau"])
            pct = tau_data.get("calibration_percentile")
            logger.info(
                f"Retrieval τ loaded: {self.retrieval_tau:.4f} "
                f"(KB-LOO p{pct} from {tau_path})"
            )
        else:
            self.retrieval_tau = None
            logger.info(
                "No retrieval_tau.json found — B1 retrieval-trust gate disabled."
            )

    def _load_kb(self, kb_dir: str) -> None:
        'English text removed.'
        vector_store = FaissVectorStore.load(kb_dir)
        metadata_path = os.path.join(kb_dir, "episodes.json")
        self.knowledge_manager = KnowledgeManager(
            vector_store=vector_store,
            metadata_path=metadata_path,
        )
        logger.info(f"KB loaded: {vector_store.count()} vectors")

        # Optional trace+log KB (built by build_kb.py): L2-normalized trace- and
        # log-only embeddings aligned with ood_episode_ids. Used as the OOD-gate
        # retrieval fallback — trace (service-agnostic topology) and log
        # (normalized templates) generalize across novel services, while the
        # metric modality collapses and drags the fused vector down. Absent →
        # fallback disabled (gate drops retrieval). The two views are concatenated
        # at query time, bypassing the in-domain-learned projector.
        self.ood_trace_vectors = None
        self.ood_log_vectors = None
        self.ood_episode_ids = None
        tv_path = os.path.join(kb_dir, "trace_vectors.npy")
        lv_path = os.path.join(kb_dir, "log_vectors.npy")
        oid_path = os.path.join(kb_dir, "ood_episode_ids.json")
        if all(os.path.isfile(p) for p in (tv_path, lv_path, oid_path)):
            self.ood_trace_vectors = np.load(tv_path)           # [N, 256], L2-normed
            self.ood_log_vectors = np.load(lv_path)             # [N, 256], L2-normed
            with open(oid_path) as f:
                self.ood_episode_ids = json.load(f)
            logger.info(f"OOD trace+log KB loaded: trace {self.ood_trace_vectors.shape}, "
                        f"log {self.ood_log_vectors.shape} (OOD-gate fallback enabled)")
        else:
            logger.info("No trace/log vectors — OOD-gate retrieval fallback disabled.")

                    

    LAYER_NAMES = ["L1", "L2", "L3", "L4"]

    @torch.no_grad()
    def encode_sample(
        self,
        sample: MultimodalSample,
    ) -> Dict[str, Any]:
        'English text removed.'
                                         
        trace_inputs = {
            "span_features": sample.span_features.unsqueeze(0).to(self.device),
            "trace_adjacency": sample.trace_adjacency.unsqueeze(0).to(self.device),
        }
        for attr in (
            "span_service_ids", "span_operation_ids",
            "span_status_classes", "span_is_root", "span_is_leaf",
            "span_is_on_critical_path",
        ):
            v = getattr(sample, attr, None)
            if v is not None:
                trace_inputs[attr] = v.unsqueeze(0).to(self.device)
        trace_emb = self.trace_encoder(trace_inputs).embeddings

        log_emb = self.log_encoder({
            "log_tokens": sample.log_tokens.unsqueeze(0).to(self.device),
            "log_mask": sample.log_attention_mask.unsqueeze(0).to(self.device),
        }).embeddings

        metric_emb, _ = self.metric_encoder(
            metric_values=sample.metric_values.unsqueeze(0).to(self.device),
            relative_time_positions=sample.relative_time_positions.unsqueeze(0).to(self.device),
        )

        fused = self.projector(log_emb, metric_emb, trace_emb)
        return {
            "fused": fused["fused"].cpu().numpy().squeeze(0),  # [512]
            "log_emb": log_emb,    # [1, 256] on device
            "metric_emb": metric_emb,  # [1, 256] on device
            "trace_emb": trace_emb,    # [1, 256] on device — for OOD trace retrieval
        }

    @torch.no_grad()
    def classify_layer(
        self,
        log_emb: torch.Tensor,
        metric_emb: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict fault layer using the standalone layer classifier.

        Returns:
            Dict with "layer", "confidence", "probs" or None if no classifier loaded.
        """
        if self.layer_classifier is None:
            return None
        logits = self.layer_classifier(log_emb, metric_emb)  # [1, 4]
        probs = torch.softmax(logits, dim=1).squeeze(0)      # [4]
        pred_idx = int(probs.argmax())
        return {
            "layer": self.LAYER_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probs": {self.LAYER_NAMES[i]: float(probs[i]) for i in range(4)},
        }

    @torch.no_grad()
    def encode_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        'English text removed.'
        trace_inputs = {
            "span_features": batch["span_features"].to(self.device),
            "trace_adjacency": batch["trace_adjacency"].to(self.device),
        }
        for k in (
            "span_service_ids", "span_operation_ids",
            "span_status_classes", "span_is_root", "span_is_leaf",
            "span_is_on_critical_path",
        ):
            if k in batch:
                trace_inputs[k] = batch[k].to(self.device)
        trace_emb = self.trace_encoder(trace_inputs).embeddings

        log_emb = self.log_encoder({
            "log_tokens": batch["log_tokens"].to(self.device),
            "log_mask": batch["log_mask"].to(self.device),
        }).embeddings

        metric_emb, _ = self.metric_encoder(
            metric_values=batch["metric_values"].to(self.device),
            relative_time_positions=batch["relative_time_positions"].to(self.device),
        )

        fused = self.projector(log_emb, metric_emb, trace_emb)
        return fused["fused"].cpu().numpy()  # [B, 512]

                    

    def retrieve_episodes(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        exclude_trace_id: Optional[str] = None,
    ) -> List[EpisodeMatch]:
        'English text removed.'
        search_k = k or self.top_k
                        
        actual_k = search_k + 1 if exclude_trace_id else search_k
        matches = self.knowledge_manager.retrieve(query_embedding, k=actual_k)

        if exclude_trace_id:
            matches = [
                m for m in matches
                if m.episode.metadata.get("trace_id") != exclude_trace_id
            ]

        return matches[:search_k]

    def _retrieve_ood(
        self,
        query_trace_emb: np.ndarray,
        query_log_emb: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """OOD-gate fallback retrieval by concatenated trace+log embedding.

        The fused vector collapses on novel services (the metric modality is
        cascade-dominated and the in-domain projector does not transfer). Trace
        (service-agnostic topology) and log (normalized templates) generalize, so
        when the gate fires we match the query's [trace‖log] vector against the
        same concatenation of the KB — bypassing the projector. Returns retrieved-
        episode dicts (same shape as the fused path); empty if no trace/log KB.
        """
        if self.ood_trace_vectors is None or self.ood_log_vectors is None:
            return []
        search_k = k or self.top_k

        def _unit(v):
            v = np.asarray(v, dtype=np.float32).squeeze()
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v

        # KB and query both: concat the two L2-normed views, then L2-norm the
        # concatenation so cosine weights trace and log equally.
        kb = np.concatenate([self.ood_trace_vectors, self.ood_log_vectors], axis=1)  # [N, 512]
        kb = kb / np.maximum(np.linalg.norm(kb, axis=1, keepdims=True), 1e-12)
        q = np.concatenate([_unit(query_trace_emb), _unit(query_log_emb)])           # [512]
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        sims = kb @ q
        topk = np.argsort(-sims)[:search_k]
        out: List[Dict[str, Any]] = []
        for idx in topk:
            ep = self.knowledge_manager.get_episode(self.ood_episode_ids[int(idx)])
            if ep is None:
                continue
            out.append({
                "episode_id": ep.id,
                "layer": ep.layer,
                "fault_type": ep.fault_type,
                "similarity": float(sims[int(idx)]),
                "description": ep.description,
            })
        return out

                      

    def summarize_telemetry(
        self,
        sample: MultimodalSample,
        raw_spans: Optional[List[Dict]] = None,
        raw_log_texts: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        'English text removed.'
        trace_summary = self._summarize_traces(sample, raw_spans)
        logs_summary = self._summarize_logs(sample, raw_log_texts)
        metrics_summary = self._summarize_metrics(sample)
        return metrics_summary, logs_summary, trace_summary

    def _summarize_traces(
        self,
        sample: MultimodalSample,
        raw_spans: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        'English text removed.'
        sf = sample.span_features  # [num_spans, 6]
        num_spans = sf.shape[0]

        summary = {
            "num_spans": num_spans,
            "error_span_count": int((sf[:, 1] > 0.5).sum().item()),
            "error_rate": f"{int((sf[:, 1] > 0.5).sum().item()) / max(num_spans, 1):.1%}",
            "entry_span_count": int((sf[:, 3] > 0.5).sum().item()),
            "exit_span_count": int((sf[:, 4] > 0.5).sum().item()),
            "local_span_count": int((sf[:, 5] > 0.5).sum().item()),
        }

                             
        if raw_spans:
                                                      
            process_map = {}
            if raw_spans and isinstance(raw_spans, list):
                                                                        
                                                          
                                                        
                pass

            services = set()
            durations = []
            error_spans_detail = []
            slow_spans_detail = []

            for span in raw_spans:
                                                     
                svc = (span.get("service_code", "")
                       or span.get("serviceCode", "")
                       or span.get("_service_name", ""))
                if svc:
                    services.add(svc)

                op = (span.get("operationName", "")
                      or span.get("operation_name", "")
                      or span.get("endpointName", ""))

                                       
                dur = span.get("duration", 0) or span.get("duration_ms", 0)
                if dur:
                    dur_ms = float(dur) / 1000.0 if dur > 10000 else float(dur)
                    durations.append(dur_ms)

                # Error detection
                is_error = span.get("is_error") or span.get("isError")
                if not is_error:
                    for tag in span.get("tags", []):
                        if tag.get("key") == "error" and tag.get("value"):
                            is_error = True
                            break
                        if tag.get("key") == "otel.status_code" and tag.get("value") == "ERROR":
                            is_error = True
                            break
                        if tag.get("key") == "http.status_code":
                            try:
                                if int(tag["value"]) >= 400:
                                    is_error = True
                                    break
                            except (ValueError, TypeError):
                                pass

                label = f"{svc}/{op}" if svc else op
                if is_error and label:
                    error_spans_detail.append(label)

            summary["services"] = sorted(services)
            if durations:
                durations_sorted = sorted(durations)
                summary["avg_duration_ms"] = f"{sum(durations) / len(durations):.1f}"
                summary["max_duration_ms"] = f"{max(durations):.1f}"
                p95_idx = min(int(len(durations_sorted) * 0.95), len(durations_sorted) - 1)
                summary["p95_duration_ms"] = f"{durations_sorted[p95_idx]:.1f}"
                                    
                avg_dur = sum(durations) / len(durations)
                for span in raw_spans:
                    dur = span.get("duration", 0) or span.get("duration_ms", 0)
                    dur_ms = float(dur) / 1000.0 if dur > 10000 else float(dur)
                    if dur_ms > avg_dur * 2 and dur_ms > 100:
                        svc = (span.get("service_code", "")
                               or span.get("serviceCode", "")
                               or span.get("_service_name", ""))
                        op = (span.get("operationName", "")
                              or span.get("operation_name", ""))
                        label = f"{svc}/{op}" if svc else op
                        if label:
                            slow_spans_detail.append(f"{label} ({dur_ms:.0f}ms)")
                if slow_spans_detail:
                    summary["slow_spans"] = slow_spans_detail[:5]
            if error_spans_detail:
                summary["error_span_details"] = error_spans_detail[:5]

        return summary

    def _summarize_logs(
        self,
        sample: MultimodalSample,
        raw_log_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        'English text removed.'
        summary: Dict[str, Any] = {
            "num_logs": sample.num_logs,
        }

        if raw_log_texts:
                               
            seen = set()
            unique_messages = []
            for msg in raw_log_texts:
                                   
                key = msg[:200]
                if key not in seen:
                    seen.add(key)
                    unique_messages.append(msg[:300])
            summary["log_messages"] = unique_messages[:10]

                    
            error_keywords = ["exception", "error", "failed", "timeout",
                              "connection refused", "oom", "killed", "crash",
                              "exhausted", "rejected", "unavailable"]
            keyword_hits = {}
            for kw in error_keywords:
                count = sum(1 for m in raw_log_texts if kw.lower() in m.lower())
                if count > 0:
                    keyword_hits[kw] = count
            if keyword_hits:
                summary["error_keyword_counts"] = keyword_hits
        else:
                            
            total_tokens = sample.log_tokens.numel()
            non_zero = int((sample.log_tokens != 0).sum().item())
            summary["non_padding_token_ratio"] = f"{non_zero / max(total_tokens, 1):.1%}"

        return summary

    # Metric index → human-readable name. Derived from _METRIC_FILES (60 entries)
    # so the order automatically matches the metric tensor channels emitted by
    # exp_runs_dataset.py. Previously a hardcoded 20-entry list misaligned with
    # the actual 60-channel tensor and starved SignalExtractor/LLM of names.
    METRIC_NAMES = [Path(f).stem for f in _METRIC_FILES]

    def _summarize_metrics(self, sample: MultimodalSample) -> Dict[str, Any]:
        'English text removed.'
        mv = sample.metric_values  # [num_metrics, seq_len]
        num_metrics = mv.shape[0]

        per_metric = []
        for i in range(num_metrics):
            vals = mv[i]
            non_zero_mask = vals != 0
            if non_zero_mask.sum() == 0:
                continue
            name = self.METRIC_NAMES[i] if i < len(self.METRIC_NAMES) else f"metric_{i}"
            per_metric.append({
                "name": name,
                "mean": f"{float(vals.mean()):.4f}",
                "std": f"{float(vals.std()):.4f}",
                "min": f"{float(vals.min()):.4f}",
                "max": f"{float(vals.max()):.4f}",
            })

                                
        stds = [float(mv[i].std()) for i in range(num_metrics) if mv[i].sum() != 0]
        mean_std = sum(stds) / len(stds) if stds else 0
        anomalous = [m for m in per_metric if float(m["std"]) > mean_std * 2]

        return {
            "num_metrics": num_metrics,
            "metric_seq_len": mv.shape[1],
            "active_metrics": len(per_metric),
            "anomalous_metric_count": len(anomalous),
            "anomalous_metrics": anomalous[:5],
            "cohort_mean_std": f"{mean_std:.4f}",
            "metric_stats": per_metric,
        }

    # ===== Retrieval Confidence =====

    @staticmethod
    def _find_tied_indices(
        retrieved: List[Dict[str, Any]],
        precision: int = 4,
    ) -> List[int]:
        """
        Find indices of episodes with the same sim score as Top-1.

        "Tied" means exactly the same value when rounded to `precision`
        decimal places. Sim score is primary — only truly identical scores
        are considered tied (FAISS ordering among ties is arbitrary).
        """
        if not retrieved:
            return []
        top1_rounded = round(retrieved[0].get("similarity", 0), precision)
        return [
            i for i, ep in enumerate(retrieved)
            if round(ep.get("similarity", 0), precision) == top1_rounded
        ]

    def _rerank_by_majority(
        self,
        retrieved: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank tied episodes by majority fault type voting.

        Only episodes with exactly the same sim score (rounded to 4dp) as
        Top-1 are considered tied. Among those, the majority fault type is
        promoted to the front. Episodes with lower sim scores are never
        reordered above higher-scored ones.

        Returns:
            Reranked list (majority type first among tied, then the rest).
        """
        if len(retrieved) < 2:
            return retrieved

        tied_indices = self._find_tied_indices(retrieved)

        if len(tied_indices) <= 1:
            return retrieved  # no ties

        # Count fault types among tied episodes
        tied_ft_counts = Counter(
            retrieved[i].get("fault_type", "") for i in tied_indices
        )
        majority_ft, majority_count = tied_ft_counts.most_common(1)[0]
        top1_ft = retrieved[0].get("fault_type", "")

        # Check plurality: majority_ft must strictly beat the runner-up
        runner_up_count = tied_ft_counts.most_common(2)[1][1] if len(tied_ft_counts) >= 2 else 0
        has_plurality = majority_count > runner_up_count

        # Only rerank if plurality type differs from current Top-1
        if majority_ft == top1_ft or not has_plurality:
            return retrieved

        # Rerank: majority type first among tied, preserve sim order within groups
        logger.info(
            f"  Majority rerank: {top1_ft} -> {majority_ft} "
            f"({majority_count}/{len(tied_indices)} tied episodes)"
        )
        tied_set = set(tied_indices)
        tied_eps = [retrieved[i] for i in tied_indices]
        non_tied_eps = [retrieved[i] for i in range(len(retrieved)) if i not in tied_set]

        majority_eps = [ep for ep in tied_eps if ep.get("fault_type") == majority_ft]
        minority_eps = [ep for ep in tied_eps if ep.get("fault_type") != majority_ft]

        return majority_eps + minority_eps + non_tied_eps

    def _assess_retrieval_confidence(
        self,
        retrieved: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assess how confident we are in the Top-1 retrieval result.

        Logic (after majority reranking):
        1. If Top-1 sim is strictly higher than Top-2 → HIGH (sim score decides)
        2. If tied: clear majority (>50%) among tied → HIGH (majority decides)
        3. If tied: no clear majority → LOW (truly ambiguous)

        Returns:
            Dict with:
                - level: "high" | "low"
                - sim_gap: similarity gap between Top-1 and Top-2
                - reason: human-readable explanation
                - competing_types: list of competing fault types (for LOW-CONF prompt)
        """
        if len(retrieved) < 2:
            return {"level": "high", "sim_gap": 1.0, "reason": "single result", "competing_types": []}

        top1_sim = retrieved[0].get("similarity", 0)
        top1_ft = retrieved[0].get("fault_type", "")
        sim_gap = top1_sim - retrieved[1].get("similarity", 0)

        # Plan A OOD gate: top-1 sim below the training-side KB-LOO threshold
        # means the query embedding is far from every KB cluster. Retrieved
        # neighbors are unreliable regardless of plurality / tie structure, so
        # signal level=low with an OOD-specific reason so the LLM stops
        # anchoring on them. Mirrors B1's design (B1 = trust retrieval over LLM
        # when sim ≥ τ; OOD gate = distrust retrieval entirely when sim < τ).
        if self.retrieval_tau is not None and top1_sim < self.retrieval_tau:
            competing = sorted({r.get("fault_type", "") for r in retrieved})
            return {
                "level": "low",
                "sim_gap": round(sim_gap, 6),
                "reason": (
                    f"OOD query: top-1 sim={top1_sim:.3f} < τ={self.retrieval_tau:.4f}. "
                    f"Retrieved cases are likely irrelevant — treat as weak background only."
                ),
                "competing_types": competing,
                "ood_gate_triggered": True,
                "top1_similarity": float(top1_sim),
                "tau": float(self.retrieval_tau),
            }

        tied_indices = self._find_tied_indices(retrieved)

        # Case 1: Top-1 has strictly higher sim than Top-2 → HIGH confidence
        if len(tied_indices) <= 1:
            return {
                "level": "high",
                "sim_gap": round(sim_gap, 6),
                "reason": f"Top-1 ({top1_ft}) has strictly higher sim",
                "competing_types": [],
            }

        # Case 2 & 3: There are ties — check plurality among tied
        tied_fts = [retrieved[i].get("fault_type", "") for i in tied_indices]
        ft_counts = Counter(tied_fts)
        majority_ft, majority_count = ft_counts.most_common(1)[0]
        runner_up_count = ft_counts.most_common(2)[1][1] if len(ft_counts) >= 2 else 0

        if majority_count > runner_up_count:
            # Clear plurality among tied → HIGH confidence
            return {
                "level": "high",
                "sim_gap": round(sim_gap, 6),
                "reason": f"Plurality vote: {majority_ft} ({majority_count}/{len(tied_fts)} tied)",
                "competing_types": [],
            }

        # True tie in votes (e.g. 2 vs 2) → LOW confidence
        competing = [ft for ft in tied_fts if ft != top1_ft]
        return {
            "level": "low",
            "sim_gap": round(sim_gap, 6),
            "reason": f"No majority among {len(tied_fts)} tied episodes: {dict(ft_counts)}",
            "competing_types": competing,
        }

                     

    def process_sample(
        self,
        sample: MultimodalSample,
        raw_spans: Optional[List[Dict]] = None,
        raw_log_texts: Optional[List[str]] = None,
        exclude_trace_id: Optional[str] = None,
    ) -> AgentContext:
        'English text removed.'
               
        enc = self.encode_sample(sample)
        embedding = enc["fused"]

        # 1b. Layer classifier prediction (independent of retrieval)
        layer_prediction = self.classify_layer(enc["log_emb"], enc["metric_emb"])

        # 2. Coarse-to-fine retrieval: restrict to predicted layer
        if layer_prediction is not None:
            matches = self.knowledge_manager.retrieve(
                embedding,
                k=self.top_k,
                layer_filter=layer_prediction["layer"],
            )
        else:
            matches = self.retrieve_episodes(
                embedding,
                exclude_trace_id=exclude_trace_id or sample.trace_id,
            )

                 
        metrics_summary, logs_summary, trace_summary = self.summarize_telemetry(
            sample, raw_spans=raw_spans, raw_log_texts=raw_log_texts,
        )

                                   
        services = trace_summary.get("services", [])

                                                                
        retrieved = []
        for m in matches:
            retrieved.append({
                "episode_id": m.episode.id,
                "layer": m.episode.layer,
                "fault_type": m.episode.fault_type,
                "similarity": m.similarity,
                "description": m.episode.description,
            })

        # 6. Majority reranking + confidence assessment
        retrieved = self._rerank_by_majority(retrieved)
        retrieval_confidence = self._assess_retrieval_confidence(retrieved)

               
        metadata = {
            "trace_id": sample.trace_id,
            "ground_truth_layer": sample.layer,
            "ground_truth_fault_type": sample.root_cause_type,
            "retrieval_confidence": retrieval_confidence,
        }
        if layer_prediction is not None:
            metadata["layer_classifier"] = layer_prediction
        if self.retrieval_tau is not None:
            metadata["retrieval_tau"] = self.retrieval_tau

        context = AgentContext(
            metrics_summary=metrics_summary,
            logs_summary=logs_summary,
            trace_summary=trace_summary,
            retrieved_episodes=retrieved,
            services_involved=services,
            metadata=metadata,
        )

        return context

    @torch.no_grad()
    def process_experiment(
        self,
        samples: List[MultimodalSample],
        detector_input: Dict[str, Any],
    ) -> AgentContext:
        """Symmetric-protocol query: encode every trace of an experiment, then
        anomaly-weighted pool to one query vector — mirroring KB's pooling so
        both sides operate on the same experiment-level granularity.

        Telemetry evidence comes from the baseline-relative evidence detector;
        ``detector_input`` carries the experiment's raw metric series, log
        entries, flattened spans, and fault/baseline windows.
        """
        if not samples:
            raise ValueError("process_experiment requires non-empty samples list")

        fused_np: List[np.ndarray] = []
        log_embs: List[torch.Tensor] = []
        metric_embs: List[torch.Tensor] = []
        trace_embs: List[torch.Tensor] = []
        scores: List[float] = []

        for sample in samples:
            enc = self.encode_sample(sample)
            fused_np.append(enc["fused"])             # np [512]
            log_embs.append(enc["log_emb"])           # tensor [1, 256]
            metric_embs.append(enc["metric_emb"])     # tensor [1, 256]
            trace_embs.append(enc["trace_emb"])       # tensor [1, 256]

            # Per-trace anomaly score — same formula as build_kb.py L199-204:
            #   error_rate  = mean(is_error column of span_features)
            #   max_latency = max(duration_norm column)
            sf = sample.span_features.cpu().numpy()
            scores.append(float(sf[:, 1].mean()) + float(sf[:, 0].max()))

        fused_arr = np.stack(fused_np, axis=0)        # [N, 512]
        log_arr = torch.cat(log_embs, dim=0)          # [N, 256]
        metric_arr = torch.cat(metric_embs, dim=0)    # [N, 256]
        trace_arr = torch.cat(trace_embs, dim=0)      # [N, 256]
        scores_arr = np.asarray(scores, dtype=np.float32)

        # Anomaly-weighted softmax-style pool — matches build_kb.py L254-265:
        floor = 0.01
        weights = np.maximum(scores_arr, floor)
        weights = weights / weights.sum()

        agg_fused = (fused_arr * weights[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(agg_fused)
        if norm > 0:
            agg_fused = agg_fused / norm

        weights_t = torch.from_numpy(weights).to(self.device)
        agg_log = (log_arr * weights_t.unsqueeze(1)).sum(dim=0, keepdim=True)        # [1, 256]
        agg_metric = (metric_arr * weights_t.unsqueeze(1)).sum(dim=0, keepdim=True)  # [1, 256]
        # Aggregated trace-only / log-only query vectors — same weights as the KB
        # side. Used (concatenated) for the OOD-gate retrieval fallback.
        agg_trace = (trace_arr * weights_t.unsqueeze(1)).sum(dim=0).cpu().numpy()    # [256]
        agg_log_np = agg_log.squeeze(0).cpu().numpy()                               # [256]

        # Representative trace = argmax(anomaly_score). LLM sees this trace's
        # real spans/logs/metrics as evidence backing the aggregated vector.
        rep_idx = int(np.argmax(scores_arr))
        rep_sample = samples[rep_idx]

        # Layer classifier on aggregated embeddings (must be trained on agg input)
        layer_prediction = self.classify_layer(agg_log, agg_metric)

        # Coarse-to-fine retrieval: restrict to the classifier's predicted layer.
        if layer_prediction is not None:
            matches = self.knowledge_manager.retrieve(
                agg_fused,
                k=self.top_k,
                layer_filter=layer_prediction["layer"],
            )
        else:
            matches = self.retrieve_episodes(agg_fused)

        # Telemetry evidence: baseline-relative scored summaries from the detector.
        metrics_summary, logs_summary, trace_summary = self.evidence_detector.detect(
            detector_input["metric_series"],
            detector_input["log_entries"],
            detector_input["raw_spans"],
            detector_input["windows"],
        )
        services = trace_summary.get("services", [])

        retrieved = [{
            "episode_id": m.episode.id,
            "layer": m.episode.layer,
            "fault_type": m.episode.fault_type,
            "similarity": m.similarity,
            "description": m.episode.description,
        } for m in matches]
        retrieved = self._rerank_by_majority(retrieved)
        # Confidence is assessed on the layer-filtered neighbors so the OOD gate
        # reflects distance to the predicted layer's KB clusters.
        retrieval_confidence = self._assess_retrieval_confidence(retrieved)

        # OOD gate: a layer-filtered top-1 similarity below τ means the query is
        # far from the predicted layer's KB clusters — the *fused* vector is OOD
        # (dragged into collapse by the non-generalizing metric/log modalities).
        # Fall back to TRACE-only retrieval: the trace embedding generalizes
        # across novel services far better, so it can still surface correct
        # same-fault-type neighbors. The layer decision stays LLM-led downstream.
        # If no trace/log KB is present, drop retrieval (LLM reasons from telemetry).
        if retrieval_confidence.get("ood_gate_triggered"):
            ood_hits = self._retrieve_ood(agg_trace, agg_log_np, k=self.top_k)
            retrieved = self._rerank_by_majority(ood_hits) if ood_hits else []
            retrieval_confidence["ood_fallback"] = (
                "trace_log_retrieval" if ood_hits else "no_retrieval"
            )

        metadata = {
            "trace_id": rep_sample.trace_id,
            "ground_truth_layer": rep_sample.layer,
            "ground_truth_fault_type": rep_sample.root_cause_type,
            "retrieval_confidence": retrieval_confidence,
            "experiment_aggregation": {
                "num_traces": len(samples),
                "rep_trace_index": rep_idx,
                "rep_anomaly_score": float(scores_arr[rep_idx]),
                "max_score": float(scores_arr.max()),
                "mean_score": float(scores_arr.mean()),
                "nonzero_ratio": float((scores_arr > floor).mean()),
            },
        }
        if layer_prediction is not None:
            metadata["layer_classifier"] = layer_prediction
        if self.retrieval_tau is not None:
            metadata["retrieval_tau"] = self.retrieval_tau

        return AgentContext(
            metrics_summary=metrics_summary,
            logs_summary=logs_summary,
            trace_summary=trace_summary,
            retrieved_episodes=retrieved,
            services_involved=services,
            metadata=metadata,
        )
