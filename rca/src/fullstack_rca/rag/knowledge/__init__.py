"""
Knowledge management for historical fault episodes.

Provides:
- Episode: Dataclass representing a fault incident
- EpisodeMatch: Retrieved episode with similarity score
- KnowledgeManager: Store and retrieve similar incidents
"""
from .episode import Episode, EpisodeMatch
from .manager import KnowledgeManager

__all__ = [
    "Episode",
    "EpisodeMatch",
    "KnowledgeManager",
]
