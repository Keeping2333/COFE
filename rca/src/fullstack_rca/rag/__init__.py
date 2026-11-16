"""
RAG (Retrieval-Augmented Generation) module for Full-stack RCA.

Provides:
- VectorStore abstractions for embedding storage
- Episode representation for historical faults
- KnowledgeManager for storing/retrieving similar incidents
"""
from .knowledge import Episode, EpisodeMatch, KnowledgeManager
from .pipeline import RCAPipeline
from .vectorstore import (
    Document,
    FilterableVectorStore,
    HybridVectorStore,
    SearchResult,
    VectorStore,
)

__all__ = [
    # Vector Store
    "VectorStore",
    "FilterableVectorStore",
    "HybridVectorStore",
    "Document",
    "SearchResult",
    # Knowledge
    "Episode",
    "EpisodeMatch",
    "KnowledgeManager",
    # Pipeline
    "RCAPipeline",
]
