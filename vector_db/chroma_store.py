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
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from llm.config import get_embedding_model

try:
    # Optional import – if llama_index Document objects are passed we can read .text and .metadata
    from llama_index.core import Document  # type: ignore
except Exception:  # pragma: no cover

    class Document:  # fallback minimal shim
        def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
            self.text = text
            self.metadata = metadata or {}


class CustomSentenceTransformerEmbedding(EmbeddingFunction[Documents]):
    """Custom embedding function that handles TensorFlow to PyTorch conversion."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the model with proper error handling and model conversion."""
        import shutil

        try:
            # Try loading the model normally
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            print(f"✓ Model loaded successfully")
        except (OSError, Exception) as e:
            error_str = str(e)
            if "from_tf=True" in error_str or "pytorch_model.bin" in error_str:
                # Model only has TensorFlow weights - need to convert or use alternative
                print(f"⚠ PyTorch weights not available for {self.model_name}")
                print(f"⚠ Converting from TensorFlow weights...")

                try:
                    # Use transformers library to load from TF and convert to PyTorch
                    from transformers import AutoModel, AutoTokenizer
                    import tempfile

                    # Load the TensorFlow model and convert
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    model = AutoModel.from_pretrained(self.model_name, from_tf=True)

                    # Save to a temporary directory as PyTorch
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model.save_pretrained(tmpdir)
                        tokenizer.save_pretrained(tmpdir)

                        # Now load with SentenceTransformer from the converted model
                        self._model = SentenceTransformer(tmpdir)

                    print(f"✓ Model converted from TensorFlow to PyTorch successfully")
                except Exception as convert_error:
                    print(f"✗ Conversion failed: {convert_error}")

                    # Clear cache and suggest alternatives
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    model_cache = os.path.join(
                        cache_dir, f"models--{self.model_name.replace('/', '--')}"
                    )
                    if os.path.exists(model_cache):
                        shutil.rmtree(model_cache, ignore_errors=True)
                        print(f"✓ Cleared cache for {self.model_name}")

                    print(
                        f"\n⚠ RECOMMENDED: Use a different embedding model with native PyTorch support:"
                    )
                    print(f"   Set environment variable: EMBEDDING_MODEL_KEY=mpnet")
                    print(f"   Or use: EMBEDDING_MODEL_KEY=bge")
                    raise RuntimeError(
                        f"Unable to load embedding model {self.model_name}. Please use a different model."
                    ) from convert_error
            else:
                raise

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of texts."""
        embeddings = self._model.encode(input, convert_to_tensor=False)
        return embeddings.tolist()


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

        print(f"Loading embedding model: {model_name}")

        # Try to use custom embedding function, fall back to ChromaDB default if it fails
        try:
            self._embedding_fn = CustomSentenceTransformerEmbedding(model_name)
        except (RuntimeError, OSError) as e:
            print(f"⚠ Failed to load {model_name}: {e}")
            print(f"⚠ Falling back to ChromaDB's default embedding function...")
            # Use ChromaDB's default embedding (all-MiniLM-L6-v2 with proper handling)
            from chromadb.utils import embedding_functions

            self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            print(f"✓ Using ChromaDB default embedding function")

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
