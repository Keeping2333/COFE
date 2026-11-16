"""Evaluate the LLM's evidence grounding — a deterministic, no-LLM check that
replaces the retired SOP-validated headline.

It reads a diagnose.py result JSON (nested schema) and, for every cited piece of
grounding evidence, cross-checks it against the same sample's telemetry_evidence:

  1. Faithfulness — fraction of cited support elements (fault-type support +
     layer support) that actually appear in the telemetry the LLM was shown.
     A high number means the grounding does not hallucinate channels/services.

  2. Confound handling — traffic-volume channels (otel_processed_*,
     http_request_rate, http_client_request_rate) only reflect load and are
     non-discriminative across all 12 fault types. We measure:
       - support-confound rate: how often such a channel is (wrongly) cited as
         SUPPORT — should be low;
       - excluded-confound recall: of those channels actually present in the
         sample's metric anomalies, how many the LLM put in
         layer_confounders_excluded — should be high.

  3. Telemetry reduction (RQ4) — how much raw telemetry (spans / logs / metric
     records the incident produced) the system collapses into the small surfaced
     evidence set an SRE actually needs to inspect. Reported as per-modality
     raw-vs-surfaced counts and an overall compression ratio. This is a PROXY for
     inspection burden, NOT a wall-clock time saving (there is no human-timing
     study — see the paper's Threats to Validity).

Usage:
    python rca/eval/eval_grounding.py results/<run>.json
    python rca/eval/eval_grounding.py results/<run>.json --output results/<run>_grounding.json
"""
import argparse
import json
import sys
from pathlib import Path


# Traffic-volume / observability-pipeline channels: they rise whenever load
# rises, so they are present for every fault type and discriminate none. A
# grounded diagnosis should exclude them, not lean on them.
def _is_confound(element: str) -> bool:
    e = (element or "").strip().lower()
    return (
        "otel_processed" in e
        or e in ("http_request_rate", "http_client_request_rate")
        or e.startswith("otel_")
    )


def _telemetry_text(sample: dict) -> str:
    """Concatenate all anomaly lines the LLM saw for this sample (lowercased)."""
    tev = sample.get("telemetry_evidence", {}) or {}
    parts = []
    for mod in ("trace", "logs", "metrics"):
        for a in (tev.get(mod, {}) or {}).get("anomalies", []) or []:
            parts.append(str(a))
    return "\n".join(parts).lower()


def _metric_channels(sample: dict) -> list:
    """The metric channel names present in this sample's anomalies (token before ':')."""
    tev = sample.get("telemetry_evidence", {}) or {}
    chans = []
    for a in (tev.get("metrics", {}) or {}).get("anomalies", []) or []:
        chans.append(str(a).split(":", 1)[0].strip())
    return chans


def _element_grounded(element: str, telemetry_text: str) -> bool:
    """A cited element is grounded if its name appears in the telemetry text.
    Elements are channel names / service.operation strings, which are long and
    specific enough that substring containment is a safe match."""
    el = (element or "").strip().lower()
    if not el:
        return False
    return el in telemetry_text


def _grade(results: list) -> dict:
    n = len(results)
    ft_cited = ft_grounded = 0
    layer_cited = layer_grounded = 0
    support_total = support_confound = 0
    excl_present = excl_caught = 0

    per_sample = []
    for r in results:
        tel = _telemetry_text(r)
        rb = r.get("reasoning", {}) or {}
        ft_sup = rb.get("supporting_evidence") or []
        ly_sup = rb.get("layer_supporting_evidence") or []
        ly_exc = rb.get("layer_confounders_excluded") or []

        s_ft_c = s_ft_g = 0
        for e in ft_sup:
            el = str(e.get("element", ""))
            s_ft_c += 1
            g = _element_grounded(el, tel)
            s_ft_g += int(g)
            support_total += 1
            support_confound += int(_is_confound(el))
        for e in ly_sup:
            el = str(e.get("element", ""))
            layer_cited += 1
            layer_grounded += int(_element_grounded(el, tel))
            support_total += 1
            support_confound += int(_is_confound(el))
        ft_cited += s_ft_c
        ft_grounded += s_ft_g

        # excluded-confound recall: confound channels present in metric anomalies
        # that the LLM actually listed under layer_confounders_excluded.
        present = {c for c in _metric_channels(r) if _is_confound(c)}
        excluded_named = " ".join(str(c.get("element", "")) for c in ly_exc).lower()
        caught = {c for c in present if c.lower() in excluded_named}
        excl_present += len(present)
        excl_caught += len(caught)

        per_sample.append({
            "run_id": r.get("run_id"),
            "ft_support_grounded": f"{s_ft_g}/{s_ft_c}",
            "confound_present": sorted(present),
            "confound_excluded": sorted(caught),
        })

    def _rate(num, den):
        return (num / den) if den else None

    cited_total = ft_cited + layer_cited
    grounded_total = ft_grounded + layer_grounded
    return {
        "num_samples": n,
        "faithfulness_overall": _rate(grounded_total, cited_total),
        "faithfulness_fault_type": _rate(ft_grounded, ft_cited),
        "faithfulness_layer": _rate(layer_grounded, layer_cited),
        "support_confound_rate": _rate(support_confound, support_total),
        "excluded_confound_recall": _rate(excl_caught, excl_present),
        "counts": {
            "cited_total": cited_total,
            "grounded_total": grounded_total,
            "support_total": support_total,
            "support_confound": support_confound,
            "confound_present_in_metrics": excl_present,
            "confound_excluded_by_llm": excl_caught,
        },
        "per_sample": per_sample,
    }


def _compression(results: list) -> dict:
    """Telemetry-reduction proxy (RQ4): raw telemetry the incident produced vs the
    small surfaced evidence set the LLM hands the SRE to inspect.

    Raw counts come from telemetry_evidence's own per-modality totals
    (num_spans / num_logs / num_metrics — the volume the detector scored over);
    surfaced counts are the lengths of each modality's `anomalies` list (plus the
    trace `services` list). Counts that the schema left as None are treated as 0
    so a missing modality never inflates the ratio. This is a PROXY for inspection
    burden, not a time measurement.
    """
    n = 0
    raw_spans = raw_logs = raw_metrics = 0
    surf_trace = surf_logs = surf_metrics = surf_services = 0
    per_sample = []
    for r in results:
        tev = r.get("telemetry_evidence", {}) or {}
        tr = tev.get("trace", {}) or {}
        lg = tev.get("logs", {}) or {}
        mt = tev.get("metrics", {}) or {}

        ns = int(tr.get("num_spans") or 0)
        nl = int(lg.get("num_logs") or 0)
        nm = int(mt.get("num_metrics") or 0)
        st = len(tr.get("anomalies", []) or [])
        sl = len(lg.get("anomalies", []) or [])
        sm = len(mt.get("anomalies", []) or [])
        sv = len(tr.get("services", []) or [])

        raw_spans += ns
        raw_logs += nl
        raw_metrics += nm
        surf_trace += st
        surf_logs += sl
        surf_metrics += sm
        surf_services += sv

        raw_i = ns + nl + nm
        surf_i = st + sl + sm
        per_sample.append({
            "run_id": r.get("run_id"),
            "raw": {"spans": ns, "logs": nl, "metrics": nm},
            "surfaced": {"trace": st, "logs": sl, "metrics": sm, "services": sv},
            "compression_ratio": (raw_i / surf_i) if surf_i else None,
        })
        n += 1

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    raw_total = raw_spans + raw_logs + raw_metrics
    surf_total = surf_trace + surf_logs + surf_metrics
    return {
        "num_samples": n,
        "compression_ratio_overall": (raw_total / surf_total) if surf_total else None,
        "compression_ratio_mean_per_sample": _mean(
            [p["compression_ratio"] for p in per_sample]
        ),
        "raw_totals": {"spans": raw_spans, "logs": raw_logs, "metrics": raw_metrics},
        "surfaced_totals": {
            "trace": surf_trace,
            "logs": surf_logs,
            "metrics": surf_metrics,
            "services": surf_services,
        },
        "surfaced_mean_per_sample": {
            "trace_anomalies": (surf_trace / n) if n else None,
            "log_templates": (surf_logs / n) if n else None,
            "metric_channels": (surf_metrics / n) if n else None,
            "services": (surf_services / n) if n else None,
        },
        "per_sample": per_sample,
    }


def _fmt(x):
    return "n/a" if x is None else f"{x:.1%}"


def _fmtf(x, suffix=""):
    return "n/a" if x is None else f"{x:.1f}{suffix}"


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate LLM evidence grounding + telemetry-reduction proxy."
    )
    ap.add_argument("result_json", help="diagnose.py result JSON (nested schema)")
    ap.add_argument("--output", default=None, help="write the grading JSON here")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text())
    results = data.get("results", data if isinstance(data, list) else [])
    if not results or "telemetry_evidence" not in (results[0] or {}):
        sys.exit("This script needs the nested diagnose.py schema "
                 "(results[i].telemetry_evidence + reasoning grounding fields).")

    g = _grade(results)
    comp = _compression(results)

    print(f"\n{'='*64}")
    print(f"EVIDENCE GROUNDING ({g['num_samples']} samples)")
    print(f"{'='*64}")
    print(f"Faithfulness (overall)        : {_fmt(g['faithfulness_overall'])}"
          f"   ({g['counts']['grounded_total']}/{g['counts']['cited_total']} cited elements exist in telemetry)")
    print(f"  - fault-type support        : {_fmt(g['faithfulness_fault_type'])}")
    print(f"  - layer support             : {_fmt(g['faithfulness_layer'])}")
    print(f"Support-confound rate (lower=better): {_fmt(g['support_confound_rate'])}"
          f"   ({g['counts']['support_confound']}/{g['counts']['support_total']} support cites are traffic confounds)")
    print(f"Excluded-confound recall (higher=better): {_fmt(g['excluded_confound_recall'])}"
          f"   ({g['counts']['confound_excluded_by_llm']}/{g['counts']['confound_present_in_metrics']} present confounds were excluded)")

    rt = comp["raw_totals"]
    st = comp["surfaced_totals"]
    sm = comp["surfaced_mean_per_sample"]
    print(f"\n{'-'*64}")
    print(f"TELEMETRY REDUCTION — proxy for inspection burden (not time)")
    print(f"{'-'*64}")
    print(f"Overall compression ratio     : {_fmtf(comp['compression_ratio_overall'], 'x')}"
          f"   (raw {rt['spans']+rt['logs']+rt['metrics']} → surfaced {st['trace']+st['logs']+st['metrics']})")
    print(f"Mean compression / sample     : {_fmtf(comp['compression_ratio_mean_per_sample'], 'x')}")
    print(f"Surfaced / sample (mean)      : "
          f"{_fmtf(sm['trace_anomalies'])} trace anomalies | "
          f"{_fmtf(sm['log_templates'])} log templates | "
          f"{_fmtf(sm['metric_channels'])} metric channels | "
          f"{_fmtf(sm['services'])} services")

    if args.output:
        out = dict(g)
        out["compression"] = comp
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote grading → {args.output}")


if __name__ == "__main__":
    main()
