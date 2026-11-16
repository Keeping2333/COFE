"""
Vector store implementations for embedding storage and retrieval.

Provides:
- VectorStore: Base abstract class
- FilterableVectorStore: With metadata filtering
- HybridVectorStore: Combined dense + sparse retrieval
- FaissVectorStore: FAISS-based local implementation
"""
from .base import (
    Document,
    FilterableVectorStore,
    HybridVectorStore,
    SearchResult,
    VectorStore,
)
from .faiss_store import FaissVectorStore

__all__ = [
    "VectorStore",
    "FilterableVectorStore",
    "HybridVectorStore",
    "FaissVectorStore",
    "Document",
    "SearchResult",
]
