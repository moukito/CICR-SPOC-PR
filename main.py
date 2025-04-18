#!/usr/bin/env python3
"""
Main entry point for the information retrieval application.

This module provides functionality to process documents, initialize language models,
create vector indices, and run an interactive query interface. It supports:
- Loading documents from files or directories
- Initializing LLM and embedding models based on configuration
- Creating vector indices from processed documents
- Running an interactive CLI for querying the document base

The application uses a combination of LlamaIndex for indexing and retrieval
and configures LLM models (Ollama) for generating responses.
"""

from .utils.initialize_models import initialize_models
from .utils.files_processor import load_documents
from .cli.interactive import InteractiveCLI

from llama_index.core import VectorStoreIndex


def main():
    """
    Main application function.

    This function:
    1. Loads documents from a specified path
    2. Initializes LLM and embedding models
    3. Creates a vector index from the documents
    4. Launches an interactive CLI for querying the document base

    Returns:
        None
    """
    documents = []
    paths = ["data/text/gps"]

    for path in paths:
        documents += load_documents(path)

    print(f"{len(documents)} documents loaded with success.")

    llm, embed_model = initialize_models("mistral", "minilm")

    # Create vector index
    print("Creating vectoriel index...")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)

    # Run interactive CLI
    cli = InteractiveCLI(query_engine)
    cli.run()


if __name__ == "__main__":
    main()
