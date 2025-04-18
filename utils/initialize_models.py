""" """

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

    # Support d'autres types de modèles à ajouter ici
    print(f"Unsupported type of embed model: {model_config['type']}")
    return None
