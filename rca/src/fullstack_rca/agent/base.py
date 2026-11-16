"""
Abstract base class for LLM-based reasoning agents.

Defines the interface for Stage 2 deep diagnosis: a single synthesis pass
that reasons over the telemetry summaries, the service call graph, and the
retrieved historical cases to produce a grounded root-cause diagnosis.

The agent:
1. Reads telemetry summaries (trace / log / metric) + retrieved cases
2. Synthesizes the root-cause layer + fault type with a confidence score
3. Grounds the diagnosis: cites the telemetry evidence that supports the
   fault type and the layer, and the present-but-irrelevant anomalies it
   excluded (traffic confounders / downstream cascades)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models.base import EncoderOutput, FaultLayer


class DiagnosisConfidence(Enum):
    """Confidence levels for diagnosis."""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5


@dataclass
class DiagnosisResult:
    """
    Complete diagnosis result from the reasoning agent.

    Attributes:
        root_cause_layer: Identified root cause layer
        root_cause_type: Specific fault type (e.g., 'cpu_pressure', 'connection_reset')
        affected_services: List of affected service names
        confidence: Overall confidence score [0, 1]
        confidence_level: Categorical confidence (HIGH/MEDIUM/LOW)
        reasoning_trace: Step-by-step reasoning trace
        recommendations: Suggested remediation actions
        supporting_evidence: telemetry elements that are the defining signature
            of the predicted fault type
        layer_supporting_evidence: telemetry elements that pin the root-cause layer
        layer_confounders_excluded: present anomalies ruled out for the layer
            (belong to another layer / traffic confounders / downstream cascades)
    """
    root_cause_layer: FaultLayer
    root_cause_type: str
    affected_services: List[str]
    confidence: float
    confidence_level: DiagnosisConfidence = DiagnosisConfidence.MEDIUM
    reasoning_trace: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    # Fault-type grounding: telemetry elements that are the DEFINING signature of
    # the predicted fault type (support only).
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    # Layer grounding: telemetry that pins the root-cause layer, plus present-but-
    # irrelevant anomalies ruled out for it (other-layer signals / traffic
    # confounders / downstream cascades).
    layer_supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    layer_confounders_excluded: List[Dict[str, Any]] = field(default_factory=list)
    # Evidence-only analysis verdict over the grounding above; a transparency
    # annotation that does NOT change the prediction.
    analysis_conclusion: str = ""

    def __post_init__(self):
        """Set confidence level based on score."""
        if self.confidence >= 0.8:
            self.confidence_level = DiagnosisConfidence.HIGH
        elif self.confidence >= 0.5:
            self.confidence_level = DiagnosisConfidence.MEDIUM
        else:
            self.confidence_level = DiagnosisConfidence.LOW

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "RootCauseLayer": self.root_cause_layer.value,
            "RootCauseType": self.root_cause_type,
            "AffectedServices": self.affected_services,
            "SupportingEvidence": self.supporting_evidence,
            "LayerSupportingEvidence": self.layer_supporting_evidence,
            "LayerConfoundersExcluded": self.layer_confounders_excluded,
            "AnalysisConclusion": self.analysis_conclusion,
            "Confidence": self.confidence,
            "ConfidenceLevel": self.confidence_level.value,
            "Reasoning": self.reasoning_trace,
            "Recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        """Convert to human-readable text."""
        lines = [
            "=" * 60,
            "DIAGNOSIS RESULT",
            "=" * 60,
            f"Root Cause Layer: {self.root_cause_layer.value}",
            f"Root Cause Type: {self.root_cause_type}",
            f"Affected Services: {', '.join(self.affected_services)}",
            f"Confidence: {self.confidence:.2%} ({self.confidence_level.value})",
            "",
            "Reasoning:",
        ]

        for i, step in enumerate(self.reasoning_trace, 1):
            lines.append(f"  {i}. {step}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


@dataclass
class AgentContext:
    """
    Context provided to the agent for diagnosis.

    Contains all information needed to diagnose an incident:
    - Encoded embeddings from each modality
    - Summaries of raw telemetry
    - Retrieved historical episodes
    - Time window and affected services

    Attributes:
        embeddings: Encoded multimodal embeddings from encoders
        metrics_summary: Summary of relevant metrics
        logs_summary: Summary of relevant logs
        trace_summary: Summary of relevant traces
        retrieved_episodes: Similar past episodes from RAG
        time_window: (start, end) timestamps of the incident
        services_involved: List of services in the incident scope
    """
    # Encoder outputs
    embeddings: Dict[str, EncoderOutput] = field(default_factory=dict)

    # Telemetry summaries (for LLM prompts)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    logs_summary: Dict[str, Any] = field(default_factory=dict)
    trace_summary: Dict[str, Any] = field(default_factory=dict)

    # RAG retrieved episodes
    retrieved_episodes: List[Dict[str, Any]] = field(default_factory=list)

    # Incident scope
    time_window: Tuple[float, float] = (0.0, 0.0)
    services_involved: List[str] = field(default_factory=list)

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """
    Definition of a tool available to the agent.

    Attributes:
        name: Tool name for function calling
        description: Tool description for LLM
        parameters_schema: JSON schema for parameters
        function: The actual callable
    """
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    function: Callable[..., str]


class Tool(ABC):
    """
    Abstract base class for agent tools.

    Tools allow the agent to query additional information
    during the reasoning process.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for function calling."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with given parameters.

        Returns:
            String result to be included in agent context
        """
        pass

    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for registration."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
            function=self.execute,
        )


class ReasoningAgent(ABC):
    """
    Abstract base class for LLM-based reasoning agents.

    Runs a single synthesis pass over the telemetry summaries, the service
    call graph, and the retrieved historical cases to produce a grounded
    root-cause diagnosis with a confidence estimate.

    Subclasses must implement:
    - synthesize_diagnosis(): produce the grounded diagnosis from context
    """

    def __init__(
        self,
        model_name: str,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize the reasoning agent.

        Args:
            model_name: LLM model identifier (e.g., 'gpt-4', 'claude-3')
            tools: List of available tools
            max_iterations: Maximum reasoning iterations
            temperature: LLM temperature for generation
        """
        self.model_name = model_name
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.max_iterations = max_iterations
        self.temperature = temperature

    @abstractmethod
    def synthesize_diagnosis(
        self,
        context: AgentContext,
    ) -> DiagnosisResult:
        """
        Synthesize the grounded root-cause diagnosis from context.

        This method should:
        1. Read the telemetry summaries + call graph + retrieved cases
        2. Identify the root-cause layer and specific fault type
        3. Ground the diagnosis (supporting / excluded evidence)
        4. Generate recommendations

        Args:
            context: Full diagnosis context

        Returns:
            Complete diagnosis result
        """
        pass

    def diagnose(self, context: AgentContext) -> DiagnosisResult:
        """
        Full diagnosis pipeline: a single grounded synthesis pass.

        Args:
            context: Diagnosis context with all telemetry

        Returns:
            Complete diagnosis result
        """
        return self.synthesize_diagnosis(context)

    def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """
        Call a tool by name.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        return self.tools[tool_name].execute(**kwargs)

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
