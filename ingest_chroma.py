"""Ingestion script for populating the Chroma vector database.

This script decouples vector store population from the interactive LLM runtime.
You can run it whenever new documents are added. The interactive CLI can then
query against the persistent Chroma collection using VECTOR_BACKEND=chroma.

Usage (Windows cmd):
    set VECTOR_BACKEND=chroma
    python -m llm.ingest_chroma --paths data/text/gps data/text/other

Environment variables:
    CHROMA_DIR          Directory where Chroma persists data (default: chroma_storage)
    CHROMA_COLLECTION   Collection name (default: documents)
    EMBEDDING_MODEL_KEY Key of embedding model from config (default: config default)

Options:
    --paths P1 P2 ...   One or more file or directory paths to ingest.
    --show-info         Print collection info after ingestion.
    --top N             (Future) placeholder for limiting number of docs.

The script is idempotent in the sense that it always creates new IDs; it does
NOT de-duplicate. Future enhancement could hash text content to avoid duplicates.
"""

from __future__ import annotations

import argparse
import os
from typing import List

from llm.utils.files_processor import load_documents
from llm.vector_db import ChromaVectorStore
from llm.config import get_embedding_model


def gather_documents(paths: List[str]):
    docs = []
    for p in paths:
        loaded = load_documents(p)
        if loaded:
            docs.extend(loaded)
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into Chroma vector DB"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="File and/or directory paths to ingest",
        required=True,
    )
    parser.add_argument(
        "--show-info", action="store_true", help="Show collection info after ingestion"
    )
    parser.add_argument(
        "--embedding-key",
        dest="embedding_key",
        help="Embedding model key (overrides EMBEDDING_MODEL_KEY env)",
    )
    args = parser.parse_args()

    persist_dir = os.getenv("CHROMA_DIR", "chroma_storage")
    collection_name = os.getenv("CHROMA_COLLECTION", "documents")
    embedding_key = args.embedding_key or os.getenv("EMBEDDING_MODEL_KEY", None)

    print(f"Ingesting into Chroma collection '{collection_name}' at '{persist_dir}'")
    if embedding_key:
        print(f"Embedding model key: {embedding_key}")
    else:
        default_embed = get_embedding_model()
        print(f"Using default embedding model: {default_embed['name']}")

    store = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_model_key=embedding_key,
    )

    documents = gather_documents(args.paths)

    if not documents:
        print("No documents gathered. Exiting.")
        return

    added = store.add_documents(documents)
    print(f"Added {added} document chunks to collection (total now: {store.count()}).")

    if args.show_info:
        print("Collection info:")
        for k, v in store.info().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":  # pragma: no cover
    main()
