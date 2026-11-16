'English text removed.'
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from ..models.base import FaultLayer
from .base import (
    AgentContext,
    DiagnosisResult,
    ReasoningAgent,
    Tool,
)

logger = logging.getLogger(__name__)

                       

SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) performing Root Cause Analysis (RCA) \
on a microservice system (train-ticket booking system).

You analyze telemetry data (traces, logs, metrics) through a 4-layer fault model:
- L1 (Infrastructure): CPU pressure, disk I/O saturation, memory pressure, host-level issues
- L2 (Network): DNS failures, HTTP aborts, network latency/loss, connection timeouts
- L3 (Platform): Pod kills/restarts, container evictions, resource quota issues
- L4 (Application): Exception injection, DB connection pool exhaustion, application errors, security check failures

You must provide structured, evidence-based analysis. Do not speculate without supporting signals.
Always respond in the exact JSON format requested."""

# Shared evidence-grounding instruction. The diagnosis (layer + fault type) is decided
# first via the retrieval ranking rule; the LLM then ties it to THIS incident's telemetry
# — citing the defining signals that SUPPORT the layer and the fault type, the anomalies
# excluded for the layer, and a final evidence-only conclusion. The conclusion is a
# transparency annotation and does NOT change the diagnosis.
EVIDENCE_RECONCILIATION_BLOCK = """\
## Evidence Analysis — ground the diagnosis like an SRE
Decide root_cause_layer + root_cause_type FIRST (per the retrieval ranking rule above),
THEN analyse the evidence. Cite ONLY elements that actually appear in the evidence above,
with their actual values; never invent channels or values, and never fabricate a
signature to make the diagnosis look stronger.

1. supporting_evidence (fault TYPE): the metric channels / log templates / trace spans
   whose observed values are the DEFINING signature of root_cause_type. Examples:
   database_connection_exhaustion -> db_connections_usage/pending/wait spikes +
   "connection pool"/SQL-error logs; cpu_pressure -> system_cpu_utilization sustained
   high; memory_pressure -> system_memory_* + OOM logs; http_error_injection -> 5xx /
   exception logs at the failing service; jvm_memory_pressure -> jvm_memory_used +
   jvm_gc_* rising; connection_reset/network faults -> connection-reset/timeout logs +
   network_* channels + services whose spans VANISHED.

2. layer_supporting_evidence (root-cause LAYER): the signals that place the root cause
   at root_cause_layer rather than another layer — e.g. L1 host-level system_* metrics;
   L2 network_* / connection-error logs; L3 kube_pod_* / pod-restart / liveness events;
   L4 application jvm_* / db_* / 5xx-exception logs.

3. layer_confounders_excluded: present anomalies that could superficially point to a
   DIFFERENT layer than root_cause_layer. For EACH, name the competing layer it might
   suggest (suggests_layer) and give the reason it does NOT move the root cause there.
   Examples: jvm_gc_*/db_* high -> might suggest L4 application, but here it is a
   downstream effect of the L1/L2 fault, not an app bug; high latency at the gateway ->
   might suggest L2 network, but it is a downstream cascade waiting on the culprit;
   traffic-volume channels (otel_processed_logs/spans, http_request_rate,
   http_client_request_rate) -> only reflect load, pin no layer.

4. analysis_conclusion: a short, purely evidence-based verdict. Weighing the supporting
   evidence against the excluded confounders, what does the telemetry ALONE indicate? If
   the dominant anomalies actually point to a different fault type / layer than the
   diagnosis, SAY SO plainly. Do NOT factor in retrieval / similarity confidence. This is
   a transparency annotation and does NOT change root_cause_layer / root_cause_type."""

# Shared output schema for both synthesis templates. Inserted as a plain value via
# str.format(), so its braces are literal (no doubling needed).
SYNTHESIS_OUTPUT_SCHEMA = """\
{
    "root_cause_layer": "L1" or "L2" or "L3" or "L4",
    "root_cause_type": "specific fault type name",
    "supporting_evidence": [
        {"modality": "metric|log|trace", "element": "channel / template / service.operation", "value": "the observed value (e.g. anomaly 473x, error 100%)", "why": "why this is the defining signature of root_cause_type"}
    ],
    "layer_supporting_evidence": [
        {"modality": "metric|log|trace", "element": "channel / template / service", "value": "the observed value", "why": "why this places the root cause at root_cause_layer"}
    ],
    "layer_confounders_excluded": [
        {"modality": "metric|log|trace", "element": "channel / service", "suggests_layer": "L2" or "L3" or "L4", "reason": "why this does NOT move the root cause to suggests_layer"}
    ],
    "analysis_conclusion": "evidence-only verdict on what the telemetry alone indicates (does NOT change the diagnosis above)",
    "affected_services": ["service1", "service2"],
    "confidence": 0.0,
    "reasoning_trace": ["step 1", "step 2"],
    "recommendations": ["recommendation 1"]
}"""

SYNTHESIS_TEMPLATE_HIGH_CONF = """\
Synthesize a final root cause diagnosis from the telemetry and retrieved cases below.

## Service-Level Trace Evidence (current incident)
{trace_summary}

## Metric Evidence (current incident)
{metrics_summary}

## Log Evidence (current incident)
{logs_summary}

## Service Call Graph & Per-Service Status
{call_graph_block}

## Similar Historical Cases (ranked by vector similarity)
{retrieved_episodes}

## IMPORTANT: Retrieval Ranking Rule
The cases above are ranked by embedding similarity from a trained multimodal encoder.
- **Case 1 (highest similarity) should be treated as the strongest prior** — default to its layer and fault type unless telemetry evidence clearly contradicts it.
- When multiple cases have very similar scores (e.g., all >0.99), prefer Case 1 over majority voting.
- Only override Case 1 if the telemetry provides strong, specific evidence pointing to a different fault type.

{evidence_reconciliation_block}

## Task
Determine the most likely root cause:
1. Start from Case 1's diagnosis as the default hypothesis for layer/fault type.
2. Check whether the telemetry supports or contradicts this hypothesis.
3. Ground the diagnosis: cite supporting_evidence (fault type), layer_supporting_evidence,
   and layer_confounders_excluded.

Respond ONLY with this JSON (no other text):
{output_schema}"""

SYNTHESIS_TEMPLATE_LOW_CONF = """\
Synthesize a final root cause diagnosis from the telemetry and retrieved cases below.

## Raw Telemetry (for disambiguation)

### Metric Data
{metrics_summary}

### Log Data
{logs_summary}

### Service-Level Trace Evidence
{trace_summary}

### Service Call Graph & Per-Service Status
{call_graph_block}

## Similar Historical Cases (ranked by vector similarity)
{retrieved_episodes}

## ⚠ LOW-CONFIDENCE RETRIEVAL WARNING
{retrieval_warning_reason}
Competing / candidate fault types: {competing_types}

**Because retrieval is unreliable for this query, you MUST use telemetry evidence as the primary signal.** Use this decision framework:

### Key Discriminating Signals
- **cpu_pressure vs memory_pressure vs disk_io_saturation (L1)**:
  - cpu_pressure → system_cpu_utilization sustained >90%, logs mention "throttling"
  - memory_pressure → system_memory_utilization >90%, logs mention "OOM"/"killed"/"memory"
  - disk_io_saturation → system_disk_io_time spikes, logs mention "I/O"/"disk"/"slow read"
- **memory_pressure vs connection_reset (L1↔L2)**:
  - memory_pressure → system_memory_usage/utilization anomalous, no connection-error logs
  - connection_reset → logs mention "connection reset"/"reset by peer"/"ECONNRESET"/"broken pipe", network_connections_by_state / network_errors anomalous
- **memory_pressure vs network_latency (L1↔L2)**:
  - memory_pressure → memory metrics spike, latency is secondary
  - network_latency → http_client_latency_avg/p99 spikes, network_io anomalous, memory is normal
- **database_connection_exhaustion vs jvm_memory_pressure (L4)**:
  - db_connection_exhaustion → db_connections_usage/pending spike, logs mention "connection pool"/"exhausted"
  - jvm_memory_pressure → jvm_memory_used spikes, jvm_gc_duration increases, logs mention "heap"/"GC"
- **jvm_memory_pressure (L4) vs memory_pressure (L1)** — do NOT collapse JVM-internal memory into host memory:
  - jvm_memory_pressure (L4) → jvm_memory_used / jvm_gc_count / jvm_gc_duration rising; the fault lives INSIDE one service's JVM. db_* / latency on dependents may dominate as cascades.
  - memory_pressure (L1) → host-level system_memory_utilization high + OOM/killed logs.
  - **Rule: if jvm_* metrics move but there is NO system_memory_* anomaly and NO OOM log, it is L4 jvm_memory_pressure, NOT L1. Never assert system_memory_*/OOM that is absent from the evidence.**

### Decision Process
1. Check which specific metrics are anomalous (listed in metric_stats above).
2. Check log error keywords — they are the strongest signal for disambiguation.
3. Use the discriminating signals above to select the correct fault type.
4. Only fall back to Case 1 if telemetry is truly inconclusive.

{evidence_reconciliation_block}

## Task
Determine the most likely root cause from the telemetry evidence, then ground it:
cite supporting_evidence (fault type), layer_supporting_evidence, and
layer_confounders_excluded.

Respond ONLY with this JSON (no other text):
{output_schema}"""

# ===== SocialNetwork prompt variants =====

SYSTEM_PROMPT_SN = """\
You are an expert Site Reliability Engineer (SRE) performing Root Cause Analysis (RCA) \
on a microservice system (DeathStarBench social-network).

You analyze telemetry data (traces, logs, metrics) through a 4-layer fault model:
- L1 (Infrastructure): host CPU pressure, host disk I/O saturation, host memory pressure
- L2 (Network): DNS resolution failure, network latency, packet loss
- L3 (Platform): container kill/restart, health/liveness endpoint block, cgroup CPU quota squeeze
- L4 (Application): container memory pressure (cgroup OOM), MongoDB connection-pool exhaustion, HTTP error injection

You must provide structured, evidence-based analysis. Do not speculate without supporting signals.
Always respond in the exact JSON format requested."""

EVIDENCE_RECONCILIATION_BLOCK_SN = """\
## Evidence Reconciliation — ground the diagnosis in THIS incident's telemetry
After choosing root_cause_layer + root_cause_type, do an evidence audit. Cite ONLY
elements that actually appear in the evidence above, with their actual values; never
invent channels or values.

1. supporting_evidence (fault TYPE): the metric channels / log templates / trace spans
   whose observed values are the DEFINING signature of root_cause_type. Examples:
   cpu_pressure -> system_cpu_utilization / system_cpu_load_1m sustained high;
   disk_io_saturation -> system_disk_io_time + container_blockio_bytes spikes, "I/O"/"disk" logs;
   memory_pressure -> system_memory_utilization high + host OOM/killed logs;
   dns_failure -> "no such host"/"name resolution"/"SERVFAIL"/"getaddrinfo" logs + downstream spans failing;
   network_latency -> elevated span latency across calls with no error logs;
   packet_loss -> network_errors / network_dropped rise + retransmit/timeout logs;
   container_kill -> one service's spans VANISH + container restart logs, container_cpu/memory drop to zero;
   health_endpoint_block -> liveness/health probe-failure logs, service unhealthy but spans not vanished;
   cgroup_quota_squeeze -> container_cpu_utilization throttled/capped while host system_cpu is normal;
   container_memory_pressure -> container_memory_usage approaching container_memory_limit / container OOM while host system_memory is normal;
   mongo_connection_exhaustion -> "connection pool"/"no connections available"/"pool exhausted"/Mongo-timeout logs at db-using services;
   http_error_injection -> 5xx / injected-exception logs at the failing service.

2. layer_supporting_evidence (root-cause LAYER): the signals that place the root cause
   at root_cause_layer rather than another layer — e.g. L1 host-level system_* metrics;
   L2 network_* / DNS / connection-error logs; L3 container restart / probe-failure /
   cgroup-throttle signals; L4 application 5xx / Mongo-pool / container-OOM logs.

3. layer_confounders_excluded: present anomalies that could superficially point to a
   DIFFERENT layer than root_cause_layer. For EACH, name the competing layer it might
   suggest (suggests_layer) and give the reason it does NOT move the root cause there.
   Examples: container_cpu throttled -> might suggest L1 host CPU, but host system_cpu is
   normal so it is L3 cgroup_quota_squeeze; high latency at the gateway -> might suggest
   L2 network, but it is a downstream cascade waiting on the culprit; otel_received_spans/
   otel_received_logs volume -> only reflect load, pin no layer."""

SYNTHESIS_TEMPLATE_LOW_CONF_SN = """\
Synthesize a final root cause diagnosis from the telemetry and retrieved cases below.

## Raw Telemetry (for disambiguation)

### Metric Data
{metrics_summary}

### Log Data
{logs_summary}

### Service-Level Trace Evidence
{trace_summary}

### Service Call Graph & Per-Service Status
{call_graph_block}

## Similar Historical Cases (ranked by vector similarity)
{retrieved_episodes}

## ⚠ LOW-CONFIDENCE RETRIEVAL WARNING
{retrieval_warning_reason}
Competing / candidate fault types: {competing_types}

**Because retrieval is unreliable for this query, you MUST use telemetry evidence as the primary signal.** Use this decision framework:

### Key Discriminating Signals
- **cpu_pressure vs memory_pressure vs disk_io_saturation (L1)**:
  - cpu_pressure → system_cpu_utilization sustained >90% / system_cpu_load_1m high
  - memory_pressure → system_memory_utilization >90%, logs mention "OOM"/"killed"/"memory"
  - disk_io_saturation → system_disk_io_time spikes / container_blockio_bytes high, logs mention "I/O"/"disk"
- **cpu_pressure (L1 host) vs cgroup_quota_squeeze (L3 container)** — do NOT collapse a container CPU throttle into host CPU:
  - cpu_pressure (L1) → host system_cpu_utilization / system_cpu_load_1m high
  - cgroup_quota_squeeze (L3) → container_cpu_utilization capped/throttled with host system_cpu NORMAL
  - **Rule: if container CPU is pinned but host system_cpu_* is normal, it is L3 cgroup_quota_squeeze, NOT L1.**
- **memory_pressure (L1 host) vs container_memory_pressure (L4 container)** — do NOT collapse container OOM into host memory:
  - memory_pressure (L1) → host system_memory_utilization high + host OOM/killed logs
  - container_memory_pressure (L4) → container_memory_usage approaching container_memory_limit / container OOM, host system_memory NORMAL
  - **Rule: if container memory hits its limit but host system_memory_* is normal, it is L4 container_memory_pressure, NOT L1.**
- **dns_failure vs network_latency vs packet_loss (L2)**:
  - dns_failure → logs mention "no such host"/"name resolution"/"SERVFAIL"/"getaddrinfo"; calls fail before connecting
  - network_latency → span latency raised across calls, no error logs, network metrics anomalous
  - packet_loss → network_errors / network_dropped rise, retransmit/timeout logs
- **container_kill (L3) vs health_endpoint_block (L3)**:
  - container_kill → a service's spans VANISH and a restart appears; container_cpu/memory drop to zero then resume
  - health_endpoint_block → liveness/health probe-failure logs; the service stays up (spans not vanished) but is marked unhealthy
- **mongo_connection_exhaustion (L4) vs generic latency**:
  - mongo_connection_exhaustion → "connection pool"/"no connections available"/"pool exhausted"/Mongo-timeout logs at db-using services
  - generic latency/cascade → raised latency with no Mongo-pool error logs

### Decision Process
1. Check which specific metrics are anomalous (listed in metric_stats above).
2. Check log error keywords — they are the strongest signal for disambiguation.
3. Use the discriminating signals above to select the correct fault type.
4. Only fall back to Case 1 if telemetry is truly inconclusive.

{evidence_reconciliation_block}

## Task
Determine the most likely root cause from the telemetry evidence, then ground it:
cite supporting_evidence (fault type), layer_supporting_evidence, and
layer_confounders_excluded.

Respond ONLY with this JSON (no other text):
{output_schema}"""

# Per-system prompt bundles. "tt" is the default and is byte-identical to the
# pre-modification behavior.
_PROMPT_SETS = {
    "tt": {
        "system_prompt": SYSTEM_PROMPT,
        "evidence_block": EVIDENCE_RECONCILIATION_BLOCK,
        "template_low": SYNTHESIS_TEMPLATE_LOW_CONF,
        "template_high": SYNTHESIS_TEMPLATE_HIGH_CONF,
    },
    "sn": {
        "system_prompt": SYSTEM_PROMPT_SN,
        "evidence_block": EVIDENCE_RECONCILIATION_BLOCK_SN,
        "template_low": SYNTHESIS_TEMPLATE_LOW_CONF_SN,
        "template_high": SYNTHESIS_TEMPLATE_HIGH_CONF,
    },
}


class LLMReasoningAgent(ReasoningAgent):
    'English text removed.'

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        system: str = "tt",
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        'English text removed.'
        super().__init__(model_name, tools, max_iterations, temperature)

        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        client_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.system = system
        _prompts = _PROMPT_SETS.get(system, _PROMPT_SETS["tt"])
        self._system_prompt = _prompts["system_prompt"]
        self._evidence_block = _prompts["evidence_block"]
        self._template_low = _prompts["template_low"]
        self._template_high = _prompts["template_high"]

        logger.info(f"LLMReasoningAgent initialized: model={model_name}")

                        

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        'English text removed.'
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"LLM call failed after {self.max_retries} attempts: {e}"
                    )
        return ""  # unreachable, but satisfies type checker

                        

    def synthesize_diagnosis(
        self,
        context: AgentContext,
    ) -> DiagnosisResult:
        'English text removed.'
        # Select template based on retrieval confidence
        retrieval_conf = context.metadata.get("retrieval_confidence", {})
        conf_level = retrieval_conf.get("level", "high")

        if conf_level == "low":
            competing = retrieval_conf.get("competing_types", [])
            warning_reason = retrieval_conf.get(
                "reason",
                "The retrieval model is uncertain about the best match.",
            )
            ood_tag = " [OOD gate]" if retrieval_conf.get("ood_gate_triggered") else ""
            logger.info(
                f"  LOW-CONFIDENCE retrieval{ood_tag}: competing types = {competing}"
            )
            user_prompt = self._template_low.format(
                retrieved_episodes=self._format_episodes(context.retrieved_episodes),
                retrieval_warning_reason=warning_reason,
                competing_types=", ".join(competing),
                metrics_summary=self._format_dict(context.metrics_summary),
                logs_summary=self._format_dict(context.logs_summary),
                trace_summary=self._format_dict(context.trace_summary),
                call_graph_block=self._format_call_graph(context.trace_summary),
                evidence_reconciliation_block=self._evidence_block,
                output_schema=SYNTHESIS_OUTPUT_SCHEMA,
            )
        else:
            user_prompt = self._template_high.format(
                trace_summary=self._format_dict(context.trace_summary),
                metrics_summary=self._format_dict(context.metrics_summary),
                logs_summary=self._format_dict(context.logs_summary),
                call_graph_block=self._format_call_graph(context.trace_summary),
                retrieved_episodes=self._format_episodes(context.retrieved_episodes),
                evidence_reconciliation_block=self._evidence_block,
                output_schema=SYNTHESIS_OUTPUT_SCHEMA,
            )

        response = self._call_llm(self._system_prompt, user_prompt)
        result = self._parse_diagnosis(response, context)

        logger.info(
            f"  Diagnosis: {result.root_cause_layer.value} / "
            f"{result.root_cause_type} (confidence={result.confidence:.2f})"
        )
        return result

                         

    def _extract_json(self, text: str) -> str:
        'English text removed.'
        # The model runs with response_format=json_object, so the whole response
        # is usually already valid JSON (including nested evidence arrays) —
        # parse it directly before falling back to regex.
        stripped = text.strip()
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
                                                  
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
                       
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def _fallback_layer(self, context: AgentContext) -> FaultLayer:
        """Pick a layer when the LLM response can't be parsed: trust the layer
        classifier first, then the top-1 retrieved neighbor, else L4."""
        clf = context.metadata.get("layer_classifier") or {}
        for candidate in (clf.get("layer"),
                           (context.retrieved_episodes or [{}])[0].get("layer")):
            if candidate:
                try:
                    return FaultLayer.from_string(candidate)
                except (ValueError, KeyError):
                    continue
        return FaultLayer.L4_APPLICATION

    def _parse_diagnosis(
        self,
        response: str,
        context: AgentContext,
    ) -> DiagnosisResult:
        'English text removed.'
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            root_layer_str = data.get("root_cause_layer", "L4")
            root_layer = FaultLayer.from_string(root_layer_str)

            return DiagnosisResult(
                root_cause_layer=root_layer,
                root_cause_type=data.get("root_cause_type", "unknown"),
                affected_services=data.get("affected_services", []),
                confidence=float(data.get("confidence", 0.5)),
                reasoning_trace=data.get("reasoning_trace", []),
                recommendations=data.get("recommendations", []),
                supporting_evidence=data.get("supporting_evidence", []) or [],
                layer_supporting_evidence=data.get("layer_supporting_evidence", []) or [],
                layer_confounders_excluded=data.get("layer_confounders_excluded", []) or [],
                analysis_conclusion=data.get("analysis_conclusion", "") or "",
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse diagnosis, using fallback: {e}")
            fallback_layer = self._fallback_layer(context)
            return DiagnosisResult(
                root_cause_layer=fallback_layer,
                root_cause_type="unknown",
                affected_services=[],
                confidence=0.0,
                reasoning_trace=[
                    f"Fallback: LLM response parsing failed ({e})",
                    f"Layer taken from classifier/retrieval: {fallback_layer.value}",
                ],
                recommendations=["Manual review recommended"],
            )

                       

    # Rendered separately via _format_call_graph (untruncated, with the diagnostic
    # framing), so they are skipped in the generic trace_summary dump to avoid
    # duplicated, list-truncated output.
    _DICT_SKIP_KEYS = ("call_graph", "service_status")

    def _format_call_graph(self, trace_summary: Dict[str, Any]) -> str:
        """Render the call graph + per-service status as service context: it lets
        the LLM tell a cause from a downstream cascade when grounding the layer
        (caller -> callee; a caller slow only because its callee is slow is a
        cascade, not a cause)."""
        status = trace_summary.get("service_status") or []
        edges = trace_summary.get("call_graph") or []
        if not status and not edges:
            return "(no trace topology available for this incident)"
        lines = []
        if status:
            lines.append("Per-service fault-window status (anomalous / vanished):")
            for s in status:
                lines.append(f"  - {s}")
        if edges:
            lines.append("Call graph (caller -> callee), fault subgraph:")
            for e in edges:
                lines.append(f"  - {e}")
        return "\n".join(lines)

    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        'English text removed.'
        if not d:
            return "(no data)"
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if key in self._DICT_SKIP_KEYS:
                continue
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value[:10]:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  - {json.dumps(item, ensure_ascii=False)}")
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    def _format_episodes(self, episodes: List[Dict[str, Any]]) -> str:
        'English text removed.'
        if not episodes:
            return "No similar historical cases found."
        lines = []
        for i, ep in enumerate(episodes[:5]):
            sim = ep.get('similarity', 0)
            rank_label = ">>> BEST MATCH <<<" if i == 0 else f"Rank #{i+1}"
            lines.append(f"Case {i+1} [{rank_label}] (similarity={sim:.4f}):")
            lines.append(f"  Layer: {ep.get('layer', 'unknown')}")
            lines.append(f"  Fault Type: {ep.get('fault_type', 'unknown')}")
            lines.append(f"  Description: {ep.get('description', '')}")
        return "\n".join(lines)
