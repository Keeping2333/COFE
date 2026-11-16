"""Baseline-relative evidence detector.

Produces the three per-modality summaries the LLM reasons over
(``metrics_summary`` / ``logs_summary`` / ``trace_summary``), but each element
is scored by how much the fault window deviates from the same incident's own
pre-fault normal window. The reference is the incident's own normal behavior,
so scoring is scale-invariant and robust on out-of-distribution fault types.

Per modality:
  * **metric** — pure statistics, no learned model. Anomaly score = robust
    mean-shift z = |fault_mean - base_mean| / (base_std + median_baseline_std),
    multiplied by the symmetric relative change. Cumulative counters are
    rate-differenced first; a channel present in the baseline but silent in the
    fault window is scored as a "vanished" anomaly (the L2 target-telemetry-dark
    case).
  * **trace** — statistical per-(service, operation) latency / error-rate
    distributions estimated from the baseline window; fault-window deviation is
    a latency z-score plus an error-rate delta.
  * **log** — statistical per-(service, pseudo-template) frequency. The
    pseudo-template is the encoder's own normalized line
    (:func:`_normalize_log_for_encoder`), so a surging or newly-appearing
    normalized line is the anomaly — no fixed keyword list, catches novel
    out-of-distribution error messages.

The returned dicts keep every key the hard-rule path emits so downstream
consumers (report builder, LLM template) stay shape-compatible; the scored
evidence is added alongside.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..data.exp_runs_dataset import _normalize_log_for_encoder
from .windows import FaultWindows

# How many flagged elements per modality to surface to the LLM. A display
# budget (mirrors the hard-rule path's [:5] cap), not a detection threshold —
# every element is scored; this only bounds how many are rendered.
TOP_N_DISPLAY = 8

_EPS = 1e-6

# A channel whose fault-window mean differs from its baseline mean by less than
# this symmetric relative change has not actually moved (the means agree to
# <1%). Its residual ratio can still spike when z-normalization amplifies a
# near-constant baseline, so it is excluded from the anomaly list rather than
# merely down-weighted. Grounded in the round_tt_ood separation: artifact
# channels (jvm_class_count / jvm_memory_limit) have rel ≤ 0.0025 while every
# genuine mover has rel ≥ 0.15.
_REL_FLOOR = 0.01

# Cumulative counters that must be rate-differenced before scoring: their raw
# value ramps monotonically in both windows, so a baseline-relative comparison
# of the raw level is meaningless. These are node-exporter / NIC counters that
# accumulate since boot (the 7 channels the blunt mask currently discards, plus
# the two error/drop NIC counters); any channel whose name ends in ``_total``
# is also a Prometheus counter by convention.
_COUNTER_CHANNELS = {
    "system_cpu_by_mode",
    "system_disk_io_bytes",
    "system_disk_operations",
    "system_disk_io_time",
    "system_disk_weighted_io_time",
    "network_io_bytes",
    "network_packets",
    "network_errors",
    "network_dropped",
}


def _is_counter(name: str) -> bool:
    return name in _COUNTER_CHANNELS or name.endswith("_total")


class EvidenceDetector:
    """Scores telemetry elements against the incident's own baseline window
    using pure statistics — no learned model.

    Each element (metric channel / span / log template) is scored by how much the
    fault window deviates from the same incident's pre-fault baseline window, so
    scoring is scale-invariant and robust on out-of-distribution faults. Stateless
    — pure NumPy, no model, no device.
    """

    # ------------------------------------------------------------------ metric

    @staticmethod
    def _rate_difference(ts: np.ndarray, vals: np.ndarray):
        """First-difference a cumulative counter into a per-interval rate.

        Negative diffs (counter resets) are clipped to 0. The diffed sample is
        timestamped at the later of the two points it spans, so window masks
        still apply.
        """
        if len(vals) < 2:
            return ts, np.zeros_like(vals)
        rate = np.clip(np.diff(vals), 0.0, None)
        return ts[1:], rate

    def _score_metric_channel(
        self, name: str, ts: np.ndarray, vals: np.ndarray, windows: FaultWindows,
    ) -> Optional[Dict[str, Any]]:
        ts, vals = prepare_channel_series(name, ts, vals)
        if len(vals) == 0:
            return None

        base_mask = windows.baseline_mask(ts)
        fault_mask = windows.fault_mask(ts)
        n_base, n_fault = int(base_mask.sum()), int(fault_mask.sum())

        # Vanished channel: observed in baseline, silent during the fault. This
        # is the L2 target-telemetry-dark signal — disappearance IS the anomaly.
        if n_base > 0 and n_fault == 0:
            return {
                "name": name,
                "score": float("inf"),
                "base_level": None,
                "fault_level": None,
                "in_fault_window": True,
                "vanished": True,
                "baseline_mean": float(vals[base_mask].mean()),
                "fault_mean": 0.0,
            }
        if n_fault == 0:
            return None

        base_vals = vals[base_mask] if n_base > 0 else vals
        base_mean = float(base_vals.mean())
        base_std = float(base_vals.std())
        fault_mean = float(vals[fault_mask].mean())

        if base_std < _EPS:
            # Constant/silent baseline → no normal fluctuation to normalize the
            # fault-window shift against; a near-zero spread would blow the ratio
            # up on any wobble. Skip such constant channels.
            return None

        # Pure-statistics baseline-relative score (no learned model). The
        # channel's own baseline std is its "typical fluctuation"; the
        # fault-window mean shift is the deviation. score_metrics() combines them
        # into a robust mean-shift z: |fault_mean - base_mean| /
        # (base_std + median_baseline_std).
        base_level = base_std                       # typical baseline fluctuation
        fault_level = abs(fault_mean - base_mean)   # fault-window drift magnitude

        # score is filled in score_metrics() with a median-fluctuation floor.
        return {
            "name": name,
            "score": None,
            "base_level": base_level,
            "fault_level": fault_level,
            "in_fault_window": True,
            "vanished": False,
            "baseline_mean": base_mean,
            "fault_mean": fault_mean,
        }

    def score_metrics(
        self, metric_series: Sequence[Any], windows: FaultWindows,
    ) -> List[Dict[str, Any]]:
        """Score every metric channel; returns flagged channels sorted by score.

        ``metric_series`` is the list of ``MetricTimeSeries`` from
        ``ExpRunsDataset._load_experiment_metrics`` (raw, pre-mask, with real
        timestamps).
        """
        scored: List[Dict[str, Any]] = []
        for series in metric_series:
            name = getattr(series, "name", None) or "unknown"
            res = self._score_metric_channel(
                name, series.timestamps, series.values, windows,
            )
            if res is not None:
                scored.append(res)

        # Per-channel mean-shift z: |fault_mean - base_mean| / base_std. Each
        # channel is normalized by its OWN baseline std, so the score is already
        # comparable across channels of wildly different raw scale (memory ~1e9
        # vs a utilization fraction ~0.5). A cross-channel floor would be in raw
        # units and would crush the small-scale channels, so there is none;
        # base_std >= _EPS is guaranteed upstream (constant baselines skipped),
        # and _EPS here is only a numerical guard.
        for d in scored:
            if d.get("vanished"):
                continue  # score already float("inf")
            ratio = d["fault_level"] / (d["base_level"] + _EPS)
            # rel ∈ [0,1]: did the value actually move? (symmetric relative
            # change). The z must be backed by a real shift — this drops a channel
            # whose tiny baseline std amplified a negligible move (rel < floor).
            b, f = d["baseline_mean"], d["fault_mean"]
            rel = abs(f - b) / (abs(f) + abs(b) + _EPS)
            d["rel"] = rel
            d["score"] = float(ratio * rel)

        # Drop near-constant channels: a fault mean within _REL_FLOOR of the
        # baseline mean means the value did not move, so any residual-ratio spike
        # is a z-norm artifact, not a real deviation. Vanished channels (rel
        # undefined, score=inf) are always kept.
        scored = [d for d in scored
                  if d.get("vanished") or d.get("rel", 1.0) >= _REL_FLOOR]

        # inf (vanished) sorts to the top; finite scores descend after.
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored

    # ------------------------------------------------------------------- trace

    @staticmethod
    def normalize_spans(raw_spans: Sequence[Dict]) -> List[Dict[str, Any]]:
        """Flatten Jaeger spans into ``{service, operation, start_s, dur_ms,
        is_error, trace_id}`` records. Mirrors the service / error extraction in
        ``RCAPipeline._summarize_traces`` but keeps ``start_s`` for windowing.
        """
        out: List[Dict[str, Any]] = []
        for span in raw_spans or []:
            svc = (span.get("service_code", "")
                   or span.get("serviceCode", "")
                   or span.get("_service_name", ""))
            op = (span.get("operationName", "")
                  or span.get("operation_name", "")
                  or span.get("endpointName", ""))
            start = span.get("startTime", 0) or 0  # Jaeger µs
            dur = span.get("duration", 0) or span.get("duration_ms", 0) or 0
            dur_ms = float(dur) / 1000.0 if dur > 10000 else float(dur)
            is_error = bool(span.get("is_error") or span.get("isError"))
            if not is_error:
                for tag in span.get("tags", []):
                    k, v = tag.get("key"), tag.get("value")
                    if k == "error" and v:
                        is_error = True
                        break
                    if k == "otel.status_code" and v == "ERROR":
                        is_error = True
                        break
                    if k == "http.status_code":
                        try:
                            if int(v) >= 400:
                                is_error = True
                                break
                        except (ValueError, TypeError):
                            pass
            out.append({
                "service": svc,
                "operation": op,
                "start_s": float(start) / 1e6 if start else 0.0,
                "dur_ms": dur_ms,
                "is_error": is_error,
                "trace_id": span.get("traceID") or span.get("traceId") or "",
            })
        return out

    @staticmethod
    def build_service_graph(raw_spans: Sequence[Dict]) -> set:
        """Caller->callee service edges from Jaeger CHILD_OF references.

        Built from all spans (the call structure is window-independent), so a
        target service whose fault-window spans VANISHED still keeps its callers
        as in-edges — the key topology cue for connection/network faults where the
        culprit's own telemetry goes dark.
        """
        sid2svc: Dict[str, str] = {}
        for sp in raw_spans or []:
            sid = sp.get("spanID") or sp.get("spanId")
            if not sid:
                continue
            sid2svc[sid] = (sp.get("_service_name") or sp.get("service_code")
                            or sp.get("serviceCode") or "")
        edges = set()
        for sp in raw_spans or []:
            csvc = sid2svc.get(sp.get("spanID") or sp.get("spanId"))
            parents = [ref.get("spanID") or ref.get("spanId")
                       for ref in (sp.get("references") or [])
                       if ref.get("refType") == "CHILD_OF"]
            pid = sp.get("parentSpanID") or sp.get("parentSpanId")
            if pid:
                parents.append(pid)
            for psid in parents:
                psvc = sid2svc.get(psid)
                if psvc and csvc and psvc != csvc:
                    edges.add((psvc, csvc))
        return edges

    def score_traces(
        self, spans: Sequence[Dict[str, Any]], windows: FaultWindows,
    ) -> List[Dict[str, Any]]:
        """Score each (service, operation) by baseline→fault latency/error shift.

        ``spans`` are the normalized records from :meth:`normalize_spans`,
        flattened across all traces of the experiment.
        """
        by_key_base: Dict[tuple, List[Dict]] = defaultdict(list)
        by_key_fault: Dict[tuple, List[Dict]] = defaultdict(list)
        for s in spans:
            t = s["start_s"]
            key = (s["service"], s["operation"])
            if windows.baseline_start <= t < windows.baseline_end:
                by_key_base[key].append(s)
            elif windows.in_fault(t):
                by_key_fault[key].append(s)

        scored: List[Dict[str, Any]] = []
        keys = set(by_key_base) | set(by_key_fault)
        for key in keys:
            svc, op = key
            base = by_key_base.get(key, [])
            fault = by_key_fault.get(key, [])
            if not fault:
                # Present in baseline, gone in the fault window — possible span
                # collapse (target service vanished / upstream stopped calling
                # it). Only flag when the baseline rate makes ≥1 occurrence
                # *expected* in the fault window (Poisson): this filters keys
                # that simply weren't re-sampled in the short window, keeping
                # genuine disappearance of frequently-seen calls.
                if base:
                    base_rate = len(base) / max(windows.baseline_duration, 1.0)
                    expected = base_rate * max(windows.fault_duration, 1.0)
                    if expected >= 1.0:
                        # Score = expected (missed) calls, so a frequently-seen
                        # call that vanishes outranks a rare one and competes on
                        # the same scale as latency/error anomalies below.
                        scored.append({
                            "service": svc, "operation": op,
                            "score": float(expected), "vanished": True,
                            "in_fault_window": True,
                            "fault_error_rate": 0.0, "example_trace_id": "",
                        })
                continue

            f_durs = np.array([s["dur_ms"] for s in fault], dtype=np.float64)
            f_err = float(np.mean([s["is_error"] for s in fault]))
            f_mean = float(f_durs.mean())
            example_tid = next((s["trace_id"] for s in fault if s["is_error"]),
                               fault[0]["trace_id"])

            if base:
                b_durs = np.array([s["dur_ms"] for s in base], dtype=np.float64)
                b_err = float(np.mean([s["is_error"] for s in base]))
                b_mean, b_std = float(b_durs.mean()), float(b_durs.std())
                if b_std < _EPS:
                    # Baseline latency was constant (e.g. service-discovery
                    # heartbeats); a near-zero variance would explode the
                    # z-score, so use relative change instead.
                    lat_z = max(0.0, f_mean - b_mean) / (abs(b_mean) + _EPS)
                else:
                    lat_z = max(0.0, (f_mean - b_mean) / (b_std + _EPS))
            else:
                # New (service, operation) in the fault window: no baseline to
                # normalize against — score on absolute error rate + latency.
                b_err = 0.0
                lat_z = f_mean / 100.0  # ms → unitless, comparable to a z-shift
            err_delta = max(0.0, f_err - b_err)
            score = lat_z + 5.0 * err_delta  # error-rate jumps weighted up

            if score > 0:
                scored.append({
                    "service": svc, "operation": op,
                    "score": float(score), "vanished": False,
                    "in_fault_window": True,
                    "fault_error_rate": f_err,
                    "fault_mean_ms": f_mean,
                    "example_trace_id": example_tid,
                })

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored

    # --------------------------------------------------------------------- log

    def score_logs(
        self, log_entries: Sequence[Dict[str, Any]], windows: FaultWindows,
    ) -> List[Dict[str, Any]]:
        """Score each (service, pseudo-template) by baseline→fault frequency
        surge. ``log_entries`` are ``{timestamp, service, message}`` from
        ``ExpRunsDataset._load_experiment_logs``.

        Counts are converted to per-second rates so the ~30s baseline and the
        ~120s fault window are comparable.
        """
        base_dur = max(windows.baseline_duration, 1.0)
        fault_dur = max(windows.fault_duration, 1.0)

        base_cnt: Dict[tuple, int] = defaultdict(int)
        fault_cnt: Dict[tuple, int] = defaultdict(int)
        example_msg: Dict[tuple, str] = {}
        for e in log_entries:
            t = e.get("timestamp", 0.0)
            tmpl = _normalize_log_for_encoder(e.get("message", ""))
            key = (e.get("service", "unknown"), tmpl)
            if windows.baseline_start <= t < windows.baseline_end:
                base_cnt[key] += 1
            elif windows.in_fault(t):
                fault_cnt[key] += 1
                example_msg.setdefault(key, e.get("message", ""))

        scored: List[Dict[str, Any]] = []
        for key, f_count in fault_cnt.items():
            svc, tmpl = key
            b_count = base_cnt.get(key, 0)
            f_rate = f_count / fault_dur
            b_rate = b_count / base_dur
            # Surge ratio; a newly-appearing template (b_rate==0) gets the full
            # fault-rate magnitude as its score.
            score = f_rate / (b_rate + _EPS)
            scored.append({
                "template": tmpl,
                "service": svc,
                "score": float(score),
                "in_fault_window": True,
                "count_fault": f_count,
                "count_base": b_count,
                "example_message": example_msg.get(key, "")[:300],
            })
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored

    # ------------------------------------------------------------------ detect

    def detect(
        self,
        metric_series: Sequence[Any],
        log_entries: Sequence[Dict[str, Any]],
        raw_spans: Sequence[Dict],
        windows: FaultWindows,
    ):
        """Score all three modalities and assemble the LLM-facing summaries.

        Returns ``(metrics_summary, logs_summary, trace_summary)`` — each dict
        carries the keys the hard-rule path emits (so downstream stays
        compatible) plus the scored evidence lists.
        """
        spans = self.normalize_spans(raw_spans)
        graph_edges = self.build_service_graph(raw_spans)

        metric_scored = self.score_metrics(metric_series, windows)
        trace_scored = self.score_traces(spans, windows)
        log_scored = self.score_logs(log_entries, windows)

        metrics_summary = self._build_metrics_summary(metric_series, metric_scored)
        trace_summary = self._build_trace_summary(spans, trace_scored, windows, graph_edges)
        logs_summary = self._build_logs_summary(log_entries, log_scored)
        return metrics_summary, logs_summary, trace_summary

    @staticmethod
    def _metric_sentence(d: Dict[str, Any]) -> str:
        name = d["name"]
        if d.get("vanished"):
            return f"{name}: VANISHED during fault (baseline mean {d['baseline_mean']:.3g})"
        return (f"{name}: anomaly {d['score']:.1f}, "
                f"fault mean {d['fault_mean']:.3g} vs baseline {d['baseline_mean']:.3g}")

    @staticmethod
    def _trace_sentence(d: Dict[str, Any]) -> str:
        label = f"{d['service']}/{d['operation']}" if d["service"] else d["operation"]
        if d.get("vanished"):
            return f"{label}: VANISHED during fault (expected ~{d['score']:.0f} calls, saw 0)"
        parts = [f"{label}: anomaly {d['score']:.1f}"]
        if d.get("fault_error_rate", 0.0) > 0:
            parts.append(f"error-rate {d['fault_error_rate']:.0%}")
        if d.get("fault_mean_ms", 0.0) > 0:
            parts.append(f"latency {d['fault_mean_ms']:.0f}ms")
        return ", ".join(parts)

    @staticmethod
    def _log_sentence(d: Dict[str, Any]) -> str:
        tmpl = d["template"][:120]
        return (f'{d["service"]}: "{tmpl}" x{d["count_fault"]} in fault '
                f'({d["count_base"]} in baseline)')

    def _build_metrics_summary(
        self, metric_series: Sequence[Any], scored: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        active = sum(1 for s in metric_series if len(getattr(s, "values", [])) > 0)
        return {
            "num_metrics": len(metric_series),
            "active_metrics": active,
            "anomaly_count": len(scored),
            "anomalies": [self._metric_sentence(d) for d in scored[:TOP_N_DISPLAY]],
            "detector": "baseline_relative",
        }

    def _build_trace_summary(
        self, spans: List[Dict[str, Any]], scored: List[Dict[str, Any]],
        windows: FaultWindows, graph_edges: Optional[set] = None,
    ) -> Dict[str, Any]:
        graph_edges = graph_edges or set()
        span_services = {s["service"] for s in spans if s["service"]}
        graph_nodes = {n for e in graph_edges for n in e}
        # Whitelist = every service seen this incident (present spans + vanished +
        # call-graph nodes), so a dark target is still a legal localization target.
        services = sorted(span_services | graph_nodes)
        fault_spans = [s for s in spans if windows.in_fault(s["start_s"])]
        err = sum(1 for s in fault_spans if s["is_error"])
        n = len(fault_spans)

        # Per-service fault-window status, aggregated from the scored spans: the
        # anomalous-or-vanished subset that localization reasons over.
        status: Dict[str, Dict[str, Any]] = {}
        for d in scored:
            svc = d.get("service")
            if not svc:
                continue
            st = status.setdefault(
                svc, {"score": 0.0, "latency_ms": 0.0, "error_rate": 0.0, "vanished": False})
            st["score"] = max(st["score"], float(d.get("score", 0.0) or 0.0))
            st["latency_ms"] = max(st["latency_ms"], float(d.get("fault_mean_ms", 0.0) or 0.0))
            st["error_rate"] = max(st["error_rate"], float(d.get("fault_error_rate", 0.0) or 0.0))
            if d.get("vanished"):
                st["vanished"] = True
        involved = set(status)
        status_sentences = []
        for svc, st in sorted(status.items(), key=lambda kv: kv[1]["score"], reverse=True)[:20]:
            bits = []
            if st["vanished"]:
                bits.append("VANISHED (inbound calls stopped during fault)")
            if st["latency_ms"] > 0:
                bits.append(f"latency {st['latency_ms']:.0f}ms")
            if st["error_rate"] > 0:
                bits.append(f"error {st['error_rate']:.0%}")
            status_sentences.append(f"{svc}: {', '.join(bits)}" if bits else svc)

        # Call graph pruned to the fault subgraph: edges touching an
        # anomalous/vanished service (keeps the prompt focused).
        sub_edges = [f"{a} -> {b}" for (a, b) in sorted(graph_edges)
                     if a in involved or b in involved]

        return {
            "num_spans": len(spans),
            "fault_window_span_count": n,
            "error_span_count": err,
            "error_rate": f"{err / max(n, 1):.1%}",
            "services": services,
            "anomalies": [self._trace_sentence(d) for d in scored[:TOP_N_DISPLAY]],
            "service_status": status_sentences,
            "call_graph": sub_edges[:40],
            "detector": "baseline_relative",
        }

    def _build_logs_summary(
        self, log_entries: Sequence[Dict[str, Any]], scored: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "num_logs": len(log_entries),
            "anomaly_count": len(scored),
            "anomalies": [self._log_sentence(d) for d in scored[:TOP_N_DISPLAY]],
            "detector": "baseline_relative",
        }


def prepare_channel_series(name: str, ts, vals):
    """Finite-filter and (for cumulative counters) rate-difference a raw metric
    channel series.

    Used by the stats detector's metric scoring
    (:meth:`EvidenceDetector._score_metric_channel`). Returns ``(ts, vals)``
    with NaN/Inf dropped and counters converted to per-interval rates; ``vals``
    may be empty.
    """
    ts = np.asarray(ts, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float32)
    finite = np.isfinite(vals)
    ts, vals = ts[finite], vals[finite]
    if len(vals) == 0:
        return ts, vals
    if _is_counter(name):
        ts, vals = EvidenceDetector._rate_difference(ts, vals)
    return ts, vals
