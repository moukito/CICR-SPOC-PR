"""Vector database backends package.

Currently includes:
- ChromaDB integration (see chroma_store.ChromaVectorStore)
- AGNO-compatible LlamaIndex wrapper (see agno_llamaindex.AgnoLlamaIndexVectorDb)

This package isolates vector storage so the LLM / reasoning layer can be
swapped or run independently from ingestion.
"""

from .chroma_store import ChromaVectorStore  # noqa: F401
from .agno_llamaindex import AgnoLlamaIndexVectorDb  # noqa: F401
