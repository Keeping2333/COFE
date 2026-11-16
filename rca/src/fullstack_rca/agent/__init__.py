"""
Agent module for LLM-based reasoning.

Provides:
- ReasoningAgent: Abstract base for diagnosis agents
- DiagnosisResult: Complete diagnosis output
- Tool abstractions for agent capabilities
"""
from .base import (
    AgentContext,
    DiagnosisConfidence,
    DiagnosisResult,
    ReasoningAgent,
    Tool,
    ToolDefinition,
)
from .llm_agent import LLMReasoningAgent

__all__ = [
    # Base
    "ReasoningAgent",
    "DiagnosisResult",
    "DiagnosisConfidence",
    "AgentContext",
    "Tool",
    "ToolDefinition",
    # LLM Agent
    "LLMReasoningAgent",
]
