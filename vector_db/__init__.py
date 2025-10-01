"""Vector database backends package.

Currently includes:
- ChromaDB integration (see chroma_store.ChromaVectorStore)

This package isolates vector storage so the LLM / reasoning layer can be
swapped or run independently from ingestion.
"""

from .chroma_store import ChromaVectorStore  # noqa: F401
