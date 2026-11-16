"""
Episode: Representation of a historical fault/anomaly incident.

An Episode encapsulates all information about a past incident:
- Multimodal embedding (fused representation)
- Text description (for LLM context)
- Ground truth (layer, fault type, affected services)
- Telemetry summaries (metrics, logs, traces)

Episodes are stored in the vector store for retrieval during
LLM-based reasoning (RAG pipeline).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TelemetrySummary:
    """
    Summary of telemetry data during an incident.

    Attributes:
        metric_summary: Key metric anomalies and their values
        log_summary: Important log patterns and error messages
        trace_summary: Trace anomalies (latency spikes, errors)
    """
    metric_summary: Dict[str, Any] = field(default_factory=dict)
    log_summary: Dict[str, Any] = field(default_factory=dict)
    trace_summary: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert summary to text for LLM context."""
        lines = []

        if self.metric_summary:
            lines.append("Metrics:")
            for metric, info in self.metric_summary.items():
                lines.append(f"  - {metric}: {info}")

        if self.log_summary:
            lines.append("Logs:")
            for pattern, count in self.log_summary.items():
                lines.append(f"  - {pattern}: {count} occurrences")

        if self.trace_summary:
            lines.append("Traces:")
            for key, value in self.trace_summary.items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)


@dataclass
class Episode:
    """
    Representation of a historical fault/anomaly incident.

    This is the core data structure for the RAG knowledge base.
    Each Episode represents one fault injection experiment or
    real-world incident with full context.

    Attributes:
        id: Unique episode identifier
        embedding: Fused multimodal embedding [embed_dim]
        layer: Root cause layer (L1/L2/L3/L4)
        fault_type: Specific fault type (e.g., 'cpu_pressure')
        affected_services: List of affected service names
        description: Human-readable description
        telemetry: Summary of relevant telemetry
        timestamp_start: Incident start time
        timestamp_end: Incident end time
        resolution: How the incident was resolved (if known)
        metadata: Additional metadata
    """
    # Identifiers
    id: str
    experiment_id: Optional[str] = None

    # Embedding for vector search
    embedding: Optional[np.ndarray] = None

    # Ground truth
    layer: str = ""  # L1, L2, L3, L4
    fault_type: str = ""  # cpu_pressure, network_latency, etc.
    affected_services: List[str] = field(default_factory=list)

    # Descriptions
    description: str = ""
    detailed_analysis: str = ""
    resolution: str = ""

    # Telemetry summary
    telemetry: Optional[TelemetrySummary] = None

    # Time window
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, include_telemetry: bool = True) -> str:
        """
        Convert episode to text description for LLM context.

        Args:
            include_telemetry: Whether to include telemetry summary

        Returns:
            Formatted text description
        """
        lines = [
            f"Episode: {self.id}",
            f"Layer: {self.layer}",
            f"Fault Type: {self.fault_type}",
            f"Affected Services: {', '.join(self.affected_services)}",
        ]

        if self.timestamp_start and self.timestamp_end:
            lines.append(f"Time Window: {self.timestamp_start} to {self.timestamp_end}")

        if self.description:
            lines.append(f"\nDescription: {self.description}")

        if self.detailed_analysis:
            lines.append(f"\nAnalysis: {self.detailed_analysis}")

        if include_telemetry and self.telemetry:
            lines.append(f"\nTelemetry:\n{self.telemetry.to_text()}")

        if self.resolution:
            lines.append(f"\nResolution: {self.resolution}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "layer": self.layer,
            "fault_type": self.fault_type,
            "affected_services": self.affected_services,
            "description": self.description,
            "detailed_analysis": self.detailed_analysis,
            "resolution": self.resolution,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "Episode":
        """Create Episode from dictionary."""
        timestamp_start = None
        timestamp_end = None

        if data.get("timestamp_start"):
            timestamp_start = datetime.fromisoformat(data["timestamp_start"])
        if data.get("timestamp_end"):
            timestamp_end = datetime.fromisoformat(data["timestamp_end"])

        return cls(
            id=data["id"],
            experiment_id=data.get("experiment_id"),
            embedding=embedding,
            layer=data.get("layer", ""),
            fault_type=data.get("fault_type", ""),
            affected_services=data.get("affected_services", []),
            description=data.get("description", ""),
            detailed_analysis=data.get("detailed_analysis", ""),
            resolution=data.get("resolution", ""),
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            metadata=data.get("metadata", {}),
        )

    def get_layer_index(self) -> int:
        """Get layer as 0-indexed integer."""
        if self.layer and self.layer.startswith("L"):
            return int(self.layer[1]) - 1
        return -1


@dataclass
class EpisodeMatch:
    """
    Result of episode retrieval from knowledge base.

    Attributes:
        episode: The matched episode
        similarity: Similarity score to query
        relevance_explanation: Why this episode is relevant
    """
    episode: Episode
    similarity: float
    relevance_explanation: str = ""

    def to_context(self) -> str:
        """Format as context for LLM prompt."""
        return (
            f"[Similar Case - Similarity: {self.similarity:.3f}]\n"
            f"{self.episode.to_text()}\n"
            f"Relevance: {self.relevance_explanation}"
        )


# Pre-defined episode templates for common fault types
EPISODE_TEMPLATES = {
    "L1_cpu_pressure": Episode(
        id="template_l1_cpu",
        layer="L1",
        fault_type="cpu_pressure",
        description="High CPU utilization on worker node causing service degradation.",
        detailed_analysis=(
            "CPU saturation typically manifests as:\n"
            "- node_cpu_seconds_total increasing rapidly\n"
            "- container_cpu_throttled_seconds_total spikes\n"
            "- Increased latency across all services on the affected node\n"
            "- Potential OOMKill events if memory pressure follows"
        ),
        resolution="Scale horizontally or vertically, optimize CPU-intensive code paths.",
    ),
    "L2_network_latency": Episode(
        id="template_l2_latency",
        layer="L2",
        fault_type="network_latency",
        description="Network latency injection causing inter-service communication delays.",
        detailed_analysis=(
            "Network latency shows:\n"
            "- Trace span durations increased by latency amount\n"
            "- http_client_requests_seconds_bucket histogram shifts right\n"
            "- Potential timeout errors in logs\n"
            "- May cascade to upstream services"
        ),
        resolution="Check network policies, service mesh configuration, or physical network.",
    ),
    "L3_pod_kill": Episode(
        id="template_l3_podkill",
        layer="L3",
        fault_type="pod_kill",
        description="Pod termination causing service unavailability.",
        detailed_analysis=(
            "Pod kill indicators:\n"
            "- Kubernetes events: 'Killing', 'Started'\n"
            "- kube_pod_container_status_restarts_total increase\n"
            "- Brief service unavailability in traces\n"
            "- Self-heals when pod restarts"
        ),
        resolution="Investigate why pod was killed, check liveness probes and resource limits.",
    ),
    "L4_db_connection": Episode(
        id="template_l4_dbconn",
        layer="L4",
        fault_type="db_connection_exhaustion",
        description="Database connection pool exhaustion causing query failures.",
        detailed_analysis=(
            "Connection exhaustion shows:\n"
            "- hikaricp_connections_pending increasing\n"
            "- 'Connection not available' in application logs\n"
            "- Database query timeouts in traces\n"
            "- Only affects services using the exhausted pool"
        ),
        resolution="Increase pool size, fix connection leaks, optimize query patterns.",
    ),
}
