"""Chroma vector store abstraction.

This module provides a thin wrapper around chromadb to:
- Initialize / load a persistent collection
- Add (ingest) documents with embeddings
- Query similar documents for a text query

It is intentionally decoupled from the LLM / agent logic so that
an offline ingestion step can populate the vector database which is
later consumed by the interactive querying runtime.
"""

from __future__ import annotations

import os
import uuid
from typing import Iterable, List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

from llm.config import get_embedding_model

try:
    # Optional import – if llama_index Document objects are passed we can read .text and .metadata
    from llama_index.core import Document  # type: ignore
except Exception:  # pragma: no cover

    class Document:  # fallback minimal shim
        def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
            self.text = text
            self.metadata = metadata or {}


class ChromaVectorStore:
    """Wrapper handling a Chroma collection lifecycle and operations."""

    def __init__(
        self,
        persist_directory: str = "chroma_storage",
        collection_name: str = "documents",
        embedding_model_key: Optional[str] = None,
    ) -> None:
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_key = embedding_model_key
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = None
        self._embedding_fn = None
        self._load_collection()

    # ---------------------- internal helpers ----------------------
    def _load_collection(self):
        if self._collection is not None:
            return
        # Resolve embedding model configuration (by key) or use default
        if self.embedding_model_key is None:
            model_conf = get_embedding_model()
        else:
            model_conf = get_embedding_model(self.embedding_model_key)
        if not model_conf:
            raise ValueError("Embedding model configuration not found.")
        model_name = model_conf["name"]
        # Use sentence-transformers embedding function (Chroma built-in) – it will download model if missing.
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, embedding_function=self._embedding_fn
        )

    # ----------------------- ingestion API -----------------------
    def add_documents(self, documents: Iterable[Document | Dict[str, Any] | str]):
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for doc in documents:
            if isinstance(doc, Document):  # llama_index Document
                text = getattr(doc, "text", "")
                metadata = getattr(doc, "metadata", {}) or {}
            elif isinstance(doc, dict):
                text = doc.get("text", "")
                metadata = doc.get("metadata", {}) or {}
            else:  # raw string
                text = str(doc)
                metadata = {}
            if not text.strip():
                continue
            texts.append(text)
            metadatas.append(metadata)
            ids.append(str(uuid.uuid4()))
        if not texts:
            return 0
        self._collection.add(documents=texts, metadatas=metadatas, ids=ids)
        return len(texts)

    # ----------------------- query API ---------------------------
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        res = self._collection.query(query_texts=[query], n_results=top_k)
        out: List[Dict[str, Any]] = []
        # Chroma returns lists inside lists (one per query)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, distances):
            out.append({"text": text, "metadata": meta, "distance": dist})
        return out

    # ----------------------- stats / utils -----------------------
    def count(self) -> int:
        return self._collection.count() if self._collection else 0

    def info(self) -> Dict[str, Any]:
        return {
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "documents": self.count(),
        }
