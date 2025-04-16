"""
Configuration settings for the information retrieval system.
"""


class ModelConfig:
    """Configuration for AI models used in the system."""

    AVAILABLE_LLM_MODELS = {
        "mistral": {
            "type": "ollama",
            "name": "mistral",
            "description": "Modèle Mistral via Ollama (local)",
        },
        "llama3": {
            "type": "ollama",
            "name": "llama3",
            "description": "Modèle Llama 3 via Ollama (local)",
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
    }

    DEFAULT_LLM = "mistral"
    DEFAULT_EMBEDDING = "minilm"


class SystemConfig:
    """
    General system configuration.
    """

    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # supported text file formats
    SUPPORTED_EXTENSIONS = {
        "text": [".txt"],
        "pdf": [".pdf"],
        "word": [".docx", ".doc"],
        "markdown": [".md"],
    }


def get_llm_model(model_key=None):
    """
    Gets the configuration of the specified LLM model.
    Uses the default model if no model is specified.
    """
    model_key = model_key or ModelConfig.DEFAULT_LLM
    return ModelConfig.AVAILABLE_LLM_MODELS.get(model_key)


def get_embedding_model(model_key=None):
    """
    Gets the configuration of the specified embedding model.
    Uses the default model if no model is specified.
    """
    model_key = model_key or ModelConfig.DEFAULT_EMBEDDING
    return ModelConfig.AVAILABLE_EMBEDDING_MODELS.get(model_key)
