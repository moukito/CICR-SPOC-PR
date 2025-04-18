"""
This module provides functions to initialize large language models (LLMs)
and embedding models based on their configurations. It supports:
- Initializing LLMs like Ollama
- Initializing embedding models like HuggingFace
- Combined initialization of both LLM and embedding models

These functions rely on configuration retrieval to determine the model type
and name, ensuring flexibility and extensibility for different model setups.
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from ..config import get_llm_model, get_embedding_model


def initialize_llm(model_key):
    """
    Initializes the Large Language Model (LLM) based on its configuration fetched
    using the provided model key. Depending on the model type, the function
    creates and returns the appropriate model instance. If the model type
    is unsupported or if the model key is invalid, it returns None.

    :param model_key: The key identifying the configuration of the LLM.
    :type model_key: Str
    :return: The instance of the initialized LLM or None if initialization fails.
    :rtype: Ollama or None
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
    Initializes an embedding model based on the provided model key.

    This function searches for a configuration corresponding to the
    given model key. If a valid model configuration is found and
    the model type is supported (e.g., HuggingFace), it initializes
    and returns the embedding model. In the case of unsupported
    model types or if no configuration is found for the specified
    key, the function prints an appropriate error message and
    returns None.

    :param model_key: The key is used to identify the embedding model
        in the configuration.
    :type model_key: Str
    :return: The initialized embedding model object if the given
        model key corresponds to a supported configuration; None
        otherwise.
    :rtype: HuggingFaceEmbedding | None
    """
    model_config = get_embedding_model(model_key)

    if not model_config:
        print(f"Error: Embed model '{model_key}' not found.")
        return None

    if model_config["type"] == "huggingface":
        return HuggingFaceEmbedding(model_name=model_config["name"])

    print(f"Unsupported type of embed model: {model_config['type']}")
    return None


def initialize_models(
    model_name: str, embed_model_name: str
) -> tuple[object, object] | tuple[None, None]:
    """
    Initialize a large language model and an embedding model.

    This function initializes both a large language model (LLM) and an embedding model using their respective names provided as arguments. If either model initialization fails, it prints a warning message indicating that the initialization is impossible and prompts verification of the configuration.

    :param model_name: Name of the large language model
    :type model_name: str

    :param embed_model_name: Name of the embedding model
    :type embed_model_name: str
    """
    llm = initialize_llm(model_name)
    embed_model = initialize_embedding(embed_model_name)

    if not llm or not embed_model:
        print(
            "Initialisation of the model impossible. Please verify the "
            "configuration."
        )
        return None, None

    return llm, embed_model
