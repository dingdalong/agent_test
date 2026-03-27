"""Backward-compatibility alias — use ChromaMemoryStore directly."""

from src.memory.chroma.store import ChromaMemoryStore as MemoryStore

__all__ = ["MemoryStore"]
