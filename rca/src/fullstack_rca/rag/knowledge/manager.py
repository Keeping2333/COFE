"""
KnowledgeManager: Manages the episode knowledge base for RAG.

Responsibilities:
1. Store and retrieve historical fault episodes
2. Build knowledge base from experiment runs
3. Provide relevant episodes for LLM reasoning
4. Maintain episode metadata and embeddings
"""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..vectorstore.base import Document, SearchResult, VectorStore
from .episode import Episode, EpisodeMatch, TelemetrySummary

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Manages the episode knowledge base for RAG-based RCA.

    The KnowledgeManager:
    1. Stores episode embeddings in a vector store
    2. Maintains episode metadata separately
    3. Provides retrieval methods for LLM context
    4. Supports building KB from experiment directories
    """

    def __init__(
        self,
        vector_store: VectorStore,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize knowledge manager.

        Args:
            vector_store: Vector store for embedding search
            metadata_path: Optional path to persist episode metadata
        """
        self.vector_store = vector_store
        self.metadata_path = Path(metadata_path) if metadata_path else None

        # Episode metadata (id -> Episode without embedding)
        self._episodes: Dict[str, Episode] = {}

        # Load existing metadata if available
        if self.metadata_path and self.metadata_path.exists():
            self._load_metadata()

    def add_episode(
        self,
        episode: Episode,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Add an episode to the knowledge base.

        Args:
            episode: Episode to add
            embedding: Optional embedding (uses episode.embedding if not provided)

        Returns:
            Episode ID
        """
        emb = embedding if embedding is not None else episode.embedding

        if emb is None:
            raise ValueError(f"No embedding provided for episode {episode.id}")

        # Ensure embedding is correct shape
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)

        # Add to vector store
        doc = Document(
            id=episode.id,
            embedding=emb.flatten(),
            metadata={
                "layer": episode.layer,
                "fault_type": episode.fault_type,
                "affected_services": episode.affected_services,
                "timestamp_start": episode.timestamp_start.timestamp() if episode.timestamp_start else 0,
            },
        )
        self.vector_store.add([doc])

        # Store metadata (without embedding to save memory)
        episode_copy = Episode(
            id=episode.id,
            experiment_id=episode.experiment_id,
            embedding=None,  # Don't store embedding in metadata
            layer=episode.layer,
            fault_type=episode.fault_type,
            affected_services=episode.affected_services,
            description=episode.description,
            detailed_analysis=episode.detailed_analysis,
            resolution=episode.resolution,
            telemetry=episode.telemetry,
            timestamp_start=episode.timestamp_start,
            timestamp_end=episode.timestamp_end,
            metadata=episode.metadata,
        )
        self._episodes[episode.id] = episode_copy

        logger.info(f"Added episode {episode.id} (layer={episode.layer})")

        return episode.id

    def add_episodes(
        self,
        episodes: List[Episode],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Add multiple episodes to the knowledge base.

        Args:
            episodes: List of episodes
            embeddings: Optional [num_episodes, embed_dim] embeddings

        Returns:
            List of episode IDs
        """
        ids = []
        for i, episode in enumerate(episodes):
            emb = embeddings[i] if embeddings is not None else None
            episode_id = self.add_episode(episode, emb)
            ids.append(episode_id)

        # Persist metadata
        if self.metadata_path:
            self._save_metadata()

        return ids

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        layer_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[EpisodeMatch]:
        """
        Retrieve similar episodes from knowledge base.

        Args:
            query_embedding: Query embedding vector
            k: Number of episodes to retrieve
            layer_filter: Optional filter by layer (e.g., "L1")
            min_similarity: Minimum similarity threshold

        Returns:
            List of EpisodeMatch objects with episodes and similarities
        """
        # Build filter expression
        filter_expr = None
        if layer_filter:
            filter_expr = f"layer == '{layer_filter}'"

        # Search vector store
        results: List[SearchResult] = self.vector_store.search(
            query_embedding=query_embedding.flatten(),
            k=k,
            filter_expr=filter_expr,
        )

        # Build matches
        matches = []
        for result in results:
            if result.score < min_similarity:
                continue

            episode = self._episodes.get(result.id)
            if episode is None:
                logger.warning(f"Episode {result.id} not found in metadata")
                continue

            match = EpisodeMatch(
                episode=episode,
                similarity=result.score,
                relevance_explanation=self._explain_relevance(episode, result),
            )
            matches.append(match)

        return matches

    def retrieve_by_layer(
        self,
        query_embedding: np.ndarray,
        layer: str,
        k: int = 3,
    ) -> List[EpisodeMatch]:
        """
        Retrieve episodes from a specific layer.

        Args:
            query_embedding: Query embedding
            layer: Layer to filter (L1, L2, L3, L4)
            k: Number of results

        Returns:
            List of matching episodes
        """
        return self.retrieve(
            query_embedding=query_embedding,
            k=k,
            layer_filter=layer,
        )

    def retrieve_all_layers(
        self,
        query_embedding: np.ndarray,
        k_per_layer: int = 2,
    ) -> Dict[str, List[EpisodeMatch]]:
        """
        Retrieve top episodes from each layer.

        Useful for comprehensive LLM context that covers all layers.

        Args:
            query_embedding: Query embedding
            k_per_layer: Number of episodes per layer

        Returns:
            Dict mapping layer -> list of matches
        """
        results = {}
        for layer in ["L1", "L2", "L3", "L4"]:
            matches = self.retrieve_by_layer(
                query_embedding=query_embedding,
                layer=layer,
                k=k_per_layer,
            )
            results[layer] = matches

        return results

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID."""
        return self._episodes.get(episode_id)

    def list_episodes(
        self,
        layer: Optional[str] = None,
        fault_type: Optional[str] = None,
    ) -> List[Episode]:
        """
        List episodes with optional filtering.

        Args:
            layer: Filter by layer
            fault_type: Filter by fault type

        Returns:
            List of matching episodes
        """
        episodes = list(self._episodes.values())

        if layer:
            episodes = [e for e in episodes if e.layer == layer]

        if fault_type:
            episodes = [e for e in episodes if e.fault_type == fault_type]

        return episodes

    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode from knowledge base.

        Args:
            episode_id: ID of episode to delete

        Returns:
            True if deleted, False if not found
        """
        if episode_id not in self._episodes:
            return False

        self.vector_store.delete([episode_id])
        del self._episodes[episode_id]

        if self.metadata_path:
            self._save_metadata()

        return True

    def build_from_experiments(
        self,
        experiment_dirs: List[Union[str, Path]],
        embedding_fn: Callable[[Path], np.ndarray],
        description_fn: Optional[Callable[[Path], str]] = None,
    ) -> int:
        """
        Build knowledge base from experiment directories.

        Args:
            experiment_dirs: List of experiment run directories
            embedding_fn: Function to compute embedding from experiment dir
            description_fn: Optional function to generate description

        Returns:
            Number of episodes added
        """
        count = 0

        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)

            if not exp_path.exists():
                logger.warning(f"Experiment directory not found: {exp_path}")
                continue

            # Load meta.json
            meta_file = exp_path / "meta.json"
            if not meta_file.exists():
                logger.warning(f"No meta.json in {exp_path}")
                continue

            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)

                # Compute embedding
                embedding = embedding_fn(exp_path)

                # Generate description
                description = ""
                if description_fn:
                    description = description_fn(exp_path)
                else:
                    description = f"Fault injection: {meta.get('anomaly_id', 'unknown')}"

                # Create episode
                episode = Episode(
                    id=meta.get("anomaly_id", exp_path.name),
                    experiment_id=exp_path.name,
                    embedding=embedding,
                    layer=meta.get("layer", ""),
                    fault_type=meta.get("root_cause_type", ""),
                    affected_services=meta.get("affected_services", []),
                    description=description,
                    metadata={"experiment_path": str(exp_path)},
                )

                self.add_episode(episode)
                count += 1

            except Exception as e:
                logger.error(f"Failed to process {exp_path}: {e}")
                continue

        logger.info(f"Built knowledge base with {count} episodes")

        return count

    def format_for_llm(
        self,
        matches: List[EpisodeMatch],
        max_episodes: int = 5,
    ) -> str:
        """
        Format retrieved episodes for LLM context.

        Args:
            matches: Retrieved episode matches
            max_episodes: Maximum number to include

        Returns:
            Formatted text for LLM prompt
        """
        if not matches:
            return "No similar historical cases found."

        lines = ["## Similar Historical Cases\n"]

        for i, match in enumerate(matches[:max_episodes]):
            lines.append(f"### Case {i+1} (Similarity: {match.similarity:.3f})")
            lines.append(match.episode.to_text(include_telemetry=False))
            lines.append("")

        return "\n".join(lines)

    def _explain_relevance(self, episode: Episode, result: SearchResult) -> str:
        """Generate relevance explanation for a match."""
        parts = []

        if episode.layer:
            parts.append(f"Same layer pattern ({episode.layer})")

        if episode.affected_services:
            parts.append(f"Involves services: {', '.join(episode.affected_services[:3])}")

        if episode.fault_type:
            parts.append(f"Fault type: {episode.fault_type}")

        return "; ".join(parts) if parts else "Embedding similarity"

    def _save_metadata(self) -> None:
        """Save episode metadata to disk."""
        if not self.metadata_path:
            return

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            episode_id: episode.to_dict()
            for episode_id, episode in self._episodes.items()
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(data)} episodes to {self.metadata_path}")

    def _load_metadata(self) -> None:
        """Load episode metadata from disk."""
        if not self.metadata_path or not self.metadata_path.exists():
            return

        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)

            for episode_id, episode_data in data.items():
                self._episodes[episode_id] = Episode.from_dict(episode_data)

            logger.info(f"Loaded {len(self._episodes)} episodes from {self.metadata_path}")

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        episodes = list(self._episodes.values())

        layer_counts = {}
        fault_type_counts = {}

        for ep in episodes:
            layer_counts[ep.layer] = layer_counts.get(ep.layer, 0) + 1
            fault_type_counts[ep.fault_type] = fault_type_counts.get(ep.fault_type, 0) + 1

        return {
            "total_episodes": len(episodes),
            "by_layer": layer_counts,
            "by_fault_type": fault_type_counts,
            "vector_store_count": self.vector_store.count(),
        }
