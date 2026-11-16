"""
Abstract base class for vector stores in the RAG pipeline.

Supports both FAISS (local) and Milvus (distributed) backends.
The vector store is used to:
1. Store embeddings of historical fault episodes
2. Retrieve similar episodes for LLM-based reasoning
3. Enable RAG (Retrieval-Augmented Generation) for RCA
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class SearchResult:
    """
    Result from vector similarity search.

    Attributes:
        id: Unique identifier of the document/episode
        score: Similarity score (higher is more similar for cosine, lower for L2)
        metadata: Associated metadata (timestamps, services, layer, etc.)
        embedding: Optional - the actual embedding vector
        text: Optional - text description if available
    """
    id: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    text: Optional[str] = None


@dataclass
class Document:
    """
    Document to be stored in the vector store.

    Attributes:
        id: Unique identifier
        embedding: Vector representation
        text: Optional text content (for hybrid search)
        metadata: Associated metadata for filtering
    """
    id: str
    embedding: np.ndarray
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class VectorStore(ABC):
    """
    Abstract interface for vector similarity search stores.

    Implementations must support:
    - Adding vectors with metadata
    - Similarity search (k-NN)
    - Filtered search (by metadata constraints)
    - Batch operations for efficiency

    Supported backends:
    - FAISS (local, fast, no external dependencies)
    - Milvus (distributed, scalable, production-ready)
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        **kwargs: Any,
    ) -> None:
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of index for search
                - "FLAT": Brute force, exact search
                - "IVF_FLAT": Inverted file index, approximate
                - "HNSW": Hierarchical Navigable Small World
            metric_type: Distance metric
                - "L2": Euclidean distance (lower = more similar)
                - "IP": Inner product (higher = more similar)
                - "COSINE": Cosine similarity (higher = more similar)
            **kwargs: Backend-specific configuration
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric_type = metric_type

    @abstractmethod
    def add(
        self,
        documents: List[Document],
        batch_size: int = 1000,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            batch_size: Batch size for insertion

        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector [embedding_dim]
            k: Number of results to return
            filter_expr: Optional filter expression
                Format depends on backend but typically:
                - "layer == 'L1'"
                - "timestamp > 1609459200"
                - "service_name in ['ts-order-service', 'ts-basic-service']"

        Returns:
            List of SearchResult ordered by similarity (best first)
        """
        pass

    @abstractmethod
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[List[SearchResult]]:
        """
        Batch search for similar vectors.

        Args:
            query_embeddings: Query vectors [num_queries, embedding_dim]
            k: Number of results per query
            filter_expr: Optional filter expression

        Returns:
            List of result lists, one per query
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    def get(self, ids: List[str]) -> List[Optional[Document]]:
        """
        Retrieve documents by ID.

        Args:
            ids: List of document IDs

        Returns:
            List of documents (None for missing IDs)
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the store."""
        pass

    @abstractmethod
    def persist(self, path: str) -> None:
        """
        Persist the index to disk.

        Args:
            path: Directory path to save the index
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs: Any) -> "VectorStore":
        """
        Load a persisted index from disk.

        Args:
            path: Directory path containing the saved index
            **kwargs: Additional configuration

        Returns:
            Loaded VectorStore instance
        """
        pass


class FilterableVectorStore(VectorStore, ABC):
    """
    Extended vector store with rich filtering capabilities.

    Supports filtering by:
    - Layer (L1, L2, L3, L4)
    - Time range
    - Service names
    - Fault types
    - Custom metadata fields
    """

    @abstractmethod
    def search_by_layer(
        self,
        query_embedding: np.ndarray,
        layer: str,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search within a specific fault layer.

        Args:
            query_embedding: Query vector
            layer: Layer filter ('L1', 'L2', 'L3', 'L4')
            k: Number of results

        Returns:
            Filtered search results
        """
        pass

    @abstractmethod
    def search_by_time_range(
        self,
        query_embedding: np.ndarray,
        start_time: float,
        end_time: float,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search within a time range.

        Args:
            query_embedding: Query vector
            start_time: Start timestamp (unix seconds)
            end_time: End timestamp (unix seconds)
            k: Number of results

        Returns:
            Time-filtered search results
        """
        pass

    @abstractmethod
    def search_by_service(
        self,
        query_embedding: np.ndarray,
        services: List[str],
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for episodes involving specific services.

        Args:
            query_embedding: Query vector
            services: List of service names to filter by
            k: Number of results

        Returns:
            Service-filtered search results
        """
        pass

    def search_by_fault_type(
        self,
        query_embedding: np.ndarray,
        fault_type: str,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for episodes with specific fault type.

        Args:
            query_embedding: Query vector
            fault_type: Fault type (e.g., 'cpu_pressure', 'network_latency')
            k: Number of results

        Returns:
            Fault-type-filtered search results
        """
        return self.search(
            query_embedding,
            k=k,
            filter_expr=f"fault_type == '{fault_type}'",
        )


class HybridVectorStore(FilterableVectorStore, ABC):
    """
    Hybrid vector store supporting both semantic and keyword search.

    Combines:
    - Dense retrieval (embedding similarity)
    - Sparse retrieval (BM25 or similar)

    Useful when:
    - Specific error messages need exact matching
    - Service names should be matched exactly
    - Combining semantic understanding with keyword precision
    """

    @abstractmethod
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        alpha: float = 0.5,
        filter_expr: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse retrieval.

        Args:
            query_embedding: Dense query vector
            query_text: Text query for sparse retrieval
            k: Number of results
            alpha: Weight for dense retrieval (1-alpha for sparse)
            filter_expr: Optional metadata filter

        Returns:
            Combined search results
        """
        pass
