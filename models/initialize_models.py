"""
This module provides unified AI model classes that inherit from BaseAIModel.
It supports:
- Initializing LLMs like Ollama
- Initializing embedding models like HuggingFace
- Combined initialization of both LLM and embedding models
- Unified interface across different vector database backends

These functions rely on configuration retrieval to determine the model type
and name, ensuring flexibility and extensibility for different model setups.
"""

from typing import Optional, List, Dict, Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.knowledge.knowledge import Knowledge

from llm.config import get_llm_model, get_embedding_model
from llm.vector_db import ChromaVectorStore
from llm.vector_db.agno_llamaindex import AgnoLlamaIndexVectorDb
from .base_ai_model import BaseAIModel

import dotenv
import os
import ollama

dotenv.load_dotenv()


class Agno(BaseAIModel):
    """Modern AI model implementation using AGNO framework with unified interface."""

    def __init__(self):
        super().__init__()
        self.db: str = "chromaDB"
        self.knowledge_base: Optional[Knowledge] = None
        self.embedding: Optional[HuggingFaceEmbedding] = None
        self.agno_embedder: Optional[HuggingfaceCustomEmbedder] = None
        self.agent: Optional[Agent] = None

    def initialize_agent(self, model_key: str) -> None:
        """Initialize the LLM agent with the specified model."""
        self.current_llm_key = model_key
        model_config = get_llm_model(model_key)

        if not model_config:
            print(f"Error: LLM model '{model_key}' not found.")
            return

        if model_config["type"] == "ollama":
            self.pull_model(model_config)

            self.agent = Agent(
                model=Ollama(id=model_config["name"]),
                knowledge=self.knowledge_base,
                debug_mode=True,
                instructions=[
                    "You have to answer the question based on the knowledge base.",
                ],
            )
            self.is_llm_initialized = True

    @staticmethod
    def pull_model(model_config):
        """
        Ensures the specified Ollama model is available locally by checking its existence and pulling (downloading) it if necessary.

        :param model_config: Configuration dictionary containing model information, must include a 'name' key for the model identifier
        :type model_config: dict
        :raises ollama.ResponseError: If an error other than 404 occurs when checking model existence
        """
        try:
            ollama.show(model_config["name"])
        except ollama.ResponseError as e:
            if e.status_code == 404:
                ollama.pull(model_config["name"])

    def initialize_embedding(self, model_key: str) -> None:
        """Initialize the embedding model with the specified configuration."""
        self.current_embedding_key = model_key
        model_config = get_embedding_model(model_key)

        if not model_config:
            print(f"Error: Embed model '{model_key}' not found.")
            return

        if model_config["type"] == "huggingface":
            # Initialize both LlamaIndex embedding (for VectorStoreIndex) and AGNO embedder
            self.embedding = HuggingFaceEmbedding(model_name=model_config["name"])
            self.agno_embedder = HuggingfaceCustomEmbedder(id=model_config["name"])
            self.is_embedding_initialized = True

    def vectorize(self, documents: List[Any]) -> None:
        """Process and vectorize documents for retrieval."""
        if not self.is_embedding_initialized:
            raise RuntimeError(
                "Embedding model not initialized. Call initialize_embedding first."
            )

        # Use LlamaIndex HuggingFaceEmbedding (compatible with VectorStoreIndex)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embedding,
        )
        retriever = VectorIndexRetriever(index)
        # Use custom wrapper that properly implements exists() method
        self.knowledge_base = Knowledge(
            vector_db=AgnoLlamaIndexVectorDb(knowledge_retriever=retriever)
        )
        self.is_vectorized = True

    def query(self, user_input: str, **kwargs) -> str:
        """Process a user query and return a response."""
        if not self.is_ready_for_query():
            raise RuntimeError("Model not ready for queries. Initialize agent first.")

        if not self.agent:
            raise RuntimeError("Agent not initialized.")

        # For AGNO framework, we can query directly
        response = self.agent.run(user_input)
        return str(response)

    def is_ready_for_query(self) -> bool:
        """Check if the model is ready to process queries."""
        return self.is_llm_initialized and self.agent is not None


class ChromaAIModel(BaseAIModel):
    """AI model variant that retrieves context from a pre-populated Chroma vector database.

    Separation of concerns:
    - Ingestion / population of Chroma is done offline (see ingest_chroma.py)
    - This runtime only connects to the existing collection and performs retrieval + LLM answer
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_model_key: str | None = None,
    ) -> None:
        super().__init__()
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_DIR", "chroma_storage"
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "documents"
        )
        self.embedding_model_key = embedding_model_key or os.getenv(
            "EMBEDDING_MODEL_KEY", None
        )
        self.vector_store = ChromaVectorStore(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_model_key=self.embedding_model_key,
        )
        self.model_name: Optional[str] = None
        # Chroma backend assumes vectorization is already done externally
        self.is_vectorized = True

    def initialize_agent(self, model_key: str) -> None:
        """Initialize the LLM agent with the specified model."""
        self.current_llm_key = model_key
        model_config = get_llm_model(model_key)
        if not model_config:
            print(f"Error: LLM model '{model_key}' not found.")
            return
        if model_config["type"] == "ollama":
            Agno.pull_model(model_config)
            self.model_name = model_config["name"]
            self.is_llm_initialized = True

    def initialize_embedding(self, model_key: str) -> None:
        """Initialize embedding - not needed for Chroma backend as embeddings are handled by Chroma."""
        self.current_embedding_key = model_key
        self.is_embedding_initialized = True
        print(
            "Embedding initialization skipped - Chroma handles embeddings internally."
        )

    def vectorize(self, documents: List[Any]) -> None:
        """Vectorization not needed for Chroma backend - documents should be ingested separately."""
        print(
            "Vectorization skipped - use ingest_chroma.py to populate the vector store."
        )
        self.is_vectorized = True

    def query(self, user_input: str, **kwargs) -> str:
        """Process a user query and return a response."""
        top_k = kwargs.get("top_k", 5)
        question = user_input
        if not self.model_name:
            raise RuntimeError("LLM not initialized – call initialize_agent first.")
        results: List[Dict[str, Any]] = self.vector_store.similarity_search(
            question, top_k=top_k
        )
        if not results:
            context = "No documents in vector store. Answer from general knowledge if possible."
        else:
            context_parts: List[str] = []
            for idx, r in enumerate(results, 1):
                snippet = r["text"]
                # Truncate very long chunks for prompt efficiency
                if len(snippet) > 1500:
                    snippet = snippet[:1500] + "..."
                context_parts.append(f"[Doc {idx}] {snippet}")
            context = "\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful assistant. Use only the provided context to answer the user's question. "
            "If the answer is not in the context, say you don't know. Keep answers concise."
        )
        from ollama import chat

        response = chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
                },
            ],
        )
        return (
            response.message.content if hasattr(response, "message") else str(response)
        )


class LlamaIndex(BaseAIModel):
    """
    A legacy implementation of an AI model interface that manages LLMs, embeddings, and knowledge bases.
    This class provides functionality for initializing large language models and
    embedding models, creating vector stores from documents, and querying the models
    with user input. It uses Ollama for LLM integration and HuggingFace for embeddings.

    Attributes:
        llm: The initialized language model instance
        knowledge_base: Vector store index containing document embeddings
        embedding: The text embedding model used for document vectorization
        agent: Query engine combining the LLM and knowledge base

    Note:
        This implementation is maintained for backward compatibility and may be
        deprecated in favor of the newer Agno class.
    """

    def __init__(self):
        super().__init__()
        self.llm = None
        self.knowledge_base = None
        self.embedding = None
        self.agent = None

    def initialize_agent(self, model_key: str) -> None:
        """Initialize the LLM agent with the specified model."""
        self.current_llm_key = model_key
        model_config = get_llm_model(model_key)

        if not model_config:
            print(f"Error: LLM model '{model_key}' not found.")
            return

        if model_config["type"] == "ollama":
            self.pull_model(model_config)
            self.llm = LlamaIndexOllama(model=model_config["name"])
            self.is_llm_initialized = True

        if self.knowledge_base is not None:
            self.agent = self.knowledge_base.as_query_engine(llm=self.llm)

    @staticmethod
    def pull_model(model_config):
        """
        Ensures the specified Ollama model is available locally by checking its existence and pulling (downloading) it if necessary.

        :param model_config: Configuration dictionary containing model information must include a 'name' key for the model identifier
        :type model_config: dict
        :raises ollama.ResponseError: If an error other than 404 occurs when checking model existence
        """
        try:
            ollama.show(model_config["name"])
        except ollama.ResponseError as e:
            if e.status_code == 404:
                ollama.pull(model_config["name"])

    def initialize_embedding(self, model_key: str) -> None:
        """Initialize the embedding model with the specified configuration."""
        self.current_embedding_key = model_key
        model_config = get_embedding_model(model_key)

        if not model_config:
            print(f"Error: Embed model '{model_key}' not found.")
            return

        if model_config["type"] == "huggingface":
            self.embedding = HuggingFaceEmbedding(model_name=model_config["name"])
            self.is_embedding_initialized = True

    def vectorize(self, documents: List[Any]) -> None:
        """Process and vectorize documents for retrieval."""
        if not self.is_embedding_initialized:
            raise RuntimeError(
                "Embedding model not initialized. Call initialize_embedding first."
            )

        self.knowledge_base = VectorStoreIndex.from_documents(
            documents, embed_model=self.embedding
        )
        self.is_vectorized = True

        # Re-create agent if LLM is already initialized
        if self.is_llm_initialized and self.llm:
            self.agent = self.knowledge_base.as_query_engine(llm=self.llm)

    def query(self, user_input: str, **kwargs) -> str:
        """Process a user query and return a response."""
        if not self.is_ready_for_query():
            raise RuntimeError(
                "Model not ready for queries. Initialize agent and vectorize documents first."
            )

        if not self.agent:
            raise RuntimeError("Agent not initialized.")

        return str(self.agent.query(user_input))

    def is_ready_for_query(self) -> bool:
        """Check if the model is ready to process queries."""
        return self.is_llm_initialized and self.agent is not None
