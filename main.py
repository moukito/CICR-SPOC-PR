#!/usr/bin/env python3
"""
Main entry point for the information retrieval application.

This module provides functionality to process documents, initialize language models,
create vector indices, and run an interactive query interface. It supports:
- Loading documents from files or directories
- Initializing LLM and embedding models based on configuration
- Creating vector indices from processed documents
- Running an interactive CLI for querying the document base

The application uses a combination of LlamaIndex for indexing and retrieval,
and configured LLM models (Ollama) for generating responses.
"""

import os
import sys

from llm.cli.interactive import InteractiveCLI
from llm.config.settings import get_llm_model, get_embedding_model
from llm.document_processor import get_document_processor

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def initialize_llm(model_key):
    """
    Initialize the LLM model according to configuration.

    Args:
        model_key (str): The key identifying the model in configuration

    Returns:
        Ollama or None: Initialized LLM model instance or None if initialization fails
    """
    model_config = get_llm_model(model_key)

    if not model_config:
        print(f"Error: LLM model '{model_key}' not found.")
        return None

    if model_config["type"] == "ollama":
        return Ollama(model=model_config["name"])

    print(f"Unsupported model type: {model_config['type']}")
    return None


def initialize_embedding(model_key):
    """
    Initialize the embedding model according to configuration.

    Args:
        model_key (str): The key identifying the embedding model in configuration

    Returns:
        HuggingFaceEmbedding or None: Initialized embedding model instance or None if initialization fails
    """
    model_config = get_embedding_model(model_key)

    if not model_config:
        print(f"Error: Embed model '{model_key}' not found.")
        return None

    if model_config["type"] == "huggingface":
        return HuggingFaceEmbedding(model_name=model_config["name"])

    # Support d'autres types de modèles à ajouter ici
    print(f"Unsupported type of embed model: {model_config['type']}")
    return None


def process_directory(directory):
    """
    Process all files in a directory and convert them to document objects.

    Args:
        directory (str): Path to the directory containing files to process

    Returns:
        list: List of document objects created from the files in the directory
    """
    documents = []

    if not os.path.isdir(directory):
        print(f"The directory {directory} do not exist.")
        return documents

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            processor = get_document_processor(file_path)
            document = processor.process_file(file_path)
            documents.append(document)
            print(f"Files succesfully loaded: {filename}")
        except Exception as e:
            print(f"An error occurred while loading {filename}: {str(e)}")

    return documents


def process_file(file_path):
    """
    Process an individual file and convert it to a document object.

    Args:
        file_path (str): Path to the file to process

    Returns:
        list: Single-element list containing the document object created from the file
    """
    processor = get_document_processor(file_path)
    document = processor.process_file(file_path)
    return [document]


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
    path = "data/text/gps"

    # Load documents
    print(f"Loading documents from {path}...")
    documents = []
    if os.path.isfile(path):
        documents = process_file(path)
    elif os.path.isdir(path):
        documents = process_directory(path)
    else:
        print(f"Invalid path: {path}")
        return

    if not documents:
        print(
            "No documents could be loaded. Please verify the path of the " "directory."
        )
        return

    print(f"{len(documents)} documents loaded with success.")

    # Initialise LLM and embedding models
    llm = initialize_llm("mistral")
    embed_model = initialize_embedding("minilm")

    if not llm or not embed_model:
        print(
            "Initialisation of the model impossible. Please verify the "
            "configuration."
        )
        return

    # Create vector index
    print("Creating vectoriel index...")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)

    # Run interactive CLI   
    cli = InteractiveCLI(query_engine)
    cli.run()


if __name__ == "__main__":
    main()
