from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from ..config import get_llm_model, get_embedding_model


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
