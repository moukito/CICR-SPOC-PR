"""
Configuration settings for the information retrieval system.

This module centralizes all configuration parameters for the application,
including AI model settings and system-wide parameters. It provides:
- Model configurations for LLMs and embedding models
- System parameters like chunk size and file format support
- Helper functions to retrieve model configurations
"""


class ModelConfig:
    """
    Configuration for AI models used in the information retrieval system.

    This class defines the available LLM and embedding models along with their
    default selections. Each model entry contains its type, name, and description
    to facilitate model selection and initialization.
    """

    AVAILABLE_LLM_MODELS = {
        "mistral": {
            "type": "ollama",
            "name": "mistral",
            "description": "Modèle Mistral via Ollama (local)",
        },
        "llama3": {
            "type": "ollama",
            "name": "llama3.1",
            "description": "Modèle Llama 3 via Ollama (local)",
        },
        "deepseek": {
            "type": "ollama",
            "name": "deepseek-r1",
            "description": "Modèle DeepSeek R1 via Ollama (local)",
        },
    }

    AVAILABLE_EMBEDDING_MODELS = {
        "minilm": {
            "type": "huggingface",
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "MiniLM L6 (light and fast)",
        },
        "mpnet": {
            "type": "huggingface",
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "MPNet (more accurate but heavier)",
        },
        "jina": {
            "type": "huggingface",
            "name": "jinaai/jina-embeddings-v2-base-code",
            "description": "Jina AI Embeddings V2 (default)",
        },
    }

    DEFAULT_LLM = "llama3"
    DEFAULT_EMBEDDING = "minilm"


class SystemConfig:
    """
    General system configuration parameters.

    This class defines system-wide settings that control the behavior of
    document processing, history management, and supported file formats.
    """

    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # supported text file formats
    SUPPORTED_EXTENSIONS = {
        "text": [".txt", ".md"],
        "pdf": [".pdf"],
        "word": [".docx", ".doc"],
    }


def get_llm_model(model_key=None):
    """
    Gets the configuration of the specified LLM model.

    Args:
        model_key (str, optional): The identifier key for the LLM model.
            Uses the default model if None.

    Returns:
        dict: Configuration dictionary for the requested LLM model containing
            type, name, and description.
    """
    model_key = model_key or ModelConfig.DEFAULT_LLM
    return ModelConfig.AVAILABLE_LLM_MODELS.get(model_key)


def get_embedding_model(model_key=None):
    """
    Gets the configuration of the specified embedding model.

    Args:
        model_key (str, optional): The identifier key for the embedding model.
            Uses the default model if None.

    Returns:
        dict: Configuration dictionary for the requested embedding model containing
            type, name, and description.
    """
    model_key = model_key or ModelConfig.DEFAULT_EMBEDDING
    return ModelConfig.AVAILABLE_EMBEDDING_MODELS.get(model_key)
