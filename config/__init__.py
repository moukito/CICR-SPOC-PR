"""
Configuration package for the information retrieval system.

This module serves as the entry point for all configuration-related functionality
in the application. It provides access to critical configuration components:

- LLM (Large Language Model) configuration access
- Embedding model configuration access

The configuration system centralizes all model settings and system parameters,
allowing other modules to easily retrieve configuration values through a clean,
consistent interface without needing to know the internal structure of the
configuration system.

Imported functions:
- get_llm_model: Retrieves configuration for language models
- get_embedding_model: Retrieves configuration for embedding models

Example usage:
    from llm.config import get_llm_model, get_embedding_model

    # Get the default LLM configuration
    llm_config = get_llm_model()
    print(f"Using LLM: {llm_config['name']} ({llm_config['description']})")

    # Get a specific embedding model configuration
    embed_config = get_embedding_model("mpnet")
    print(f"Model path: {embed_config['name']}")
"""

from .settings import get_llm_model, get_embedding_model
