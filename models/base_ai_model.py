"""
Base AI Model interface for unified model management.

This module defines the abstract base class that all AI model implementations
should inherit from to ensure consistent interfaces and behavior across
different backends (LlamaIndex, Chroma, Legacy, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any


class BaseAIModel(ABC):
    """
    Abstract base class defining the unified interface for all AI model implementations.

    This ensures consistent method signatures and behavior across different
    vector database backends and LLM integrations.
    """

    def __init__(self):
        """Initialize the AI model with common attributes."""
        self.current_llm_key: Optional[str] = None
        self.current_embedding_key: Optional[str] = None
        self.is_llm_initialized: bool = False
        self.is_embedding_initialized: bool = False
        self.is_vectorized: bool = False

    @abstractmethod
    def initialize_agent(self, model_key: str) -> None:
        """
        Initialize the LLM agent with the specified model.

        Args:
            model_key: Configuration key for the LLM model
        """
        pass

    @abstractmethod
    def initialize_embedding(self, model_key: str) -> None:
        """
        Initialize the embedding model with the specified configuration.

        Args:
            model_key: Configuration key for the embedding model
        """
        pass

    @abstractmethod
    def vectorize(self, documents: List[Any]) -> None:
        """
        Process and vectorize documents for retrieval.

        Args:
            documents: List of documents to vectorize
        """
        pass

    @abstractmethod
    def query(self, user_input: str, **kwargs) -> str:
        """
        Process a user query and return a response.

        Args:
            user_input: The user's question or query
            **kwargs: Additional parameters (e.g., top_k for retrieval)

        Returns:
            The AI model's response as a string
        """
        pass

    def is_ready_for_query(self) -> bool:
        """
        Check if the model is ready to process queries.

        Returns:
            True if the model can process queries, False otherwise
        """
        return self.is_llm_initialized

    def get_status(self) -> dict:
        """
        Get the current status of the AI model.

        Returns:
            Dictionary containing model status information
        """
        return {
            "llm_initialized": self.is_llm_initialized,
            "embedding_initialized": self.is_embedding_initialized,
            "vectorized": self.is_vectorized,
            "current_llm": self.current_llm_key,
            "current_embedding": self.current_embedding_key,
        }

    def reset(self) -> None:
        """Reset the model to initial state."""
        self.current_llm_key = None
        self.current_embedding_key = None
        self.is_llm_initialized = False
        self.is_embedding_initialized = False
        self.is_vectorized = False
