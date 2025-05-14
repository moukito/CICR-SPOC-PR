"""
Interactive Command-Line Interface for LLM-based Information Retrieval System.

This module provides a text-based user interface for interacting with the
document retrieval and question-answering system. It handles user input,
processes commands, and manages the interaction between the user and the
underlying query engine.

Features:
- Interactive question-answering with indexed documents
- Special commands for system control (help, quit, etc.)
- Model selection and configuration
- Question history tracking
- Terminal-based user interface with clear formatting

The primary class, InteractiveCLI, manages the interactive session and
processes both natural language questions and special commands prefixed
with a slash (/).

Usage:
    cli = InteractiveCLI(query_engine)
    cli.run()

Dependencies:
    - config.settings: For model configuration and selection
"""

import concurrent.futures
import os
import threading
from threading import Thread
from typing import Dict, List, Callable, Optional
from ollama import chat, ChatResponse

from llama_index.core import VectorStoreIndex

from ..config.settings import get_llm_model, get_embedding_model, ModelConfig
from ..utils.files_processor import load_documents
from ..utils.initialize_models import (
    initialize_models,
    initialize_llm,
    initialize_embedding,
)

# Definition of special commands
SPECIAL_COMMANDS = {
    "/quit": "Exit the application",
    "/exit": "Exit the application",
    "/help": "Display help and available commands",
    "/models": "Display and modify the models used",
    "/config": "Display and modify the configuration",
    "/clear": "Clear the screen",
    "/history": "Display question history",
}


class InteractiveCLI:
    """
    Interactive command-line interface for the LLM-based Information Retrieval System.

    This class manages the interactive session between the user and the system,
    handling both natural language questions and special commands. It provides
    functionality for querying documents, managing model configurations,
    tracking question history, and displaying system information.

    Attributes:
        query_engine: The engine used to process queries against indexed documents
        current_llm (str): The currently selected LLM model
        current_embedding (str): The currently selected embedding model
        question_history (list): A list of previously asked questions
        running (bool): Flag indicating if the interface is active
    """

    def __init__(self):
        """
        Initialize the interactive CLI interface with configuration settings.

        Sets up the interface with default model configurations, initializes
        the question history tracking, and connects to the provided query engine
        for processing document-based queries.

        Args:
            query_engine: The query engine instance used to process questions
                          against indexed documents. If None, queries will prompt
                          for document loading.
        """
        self.documents = []
        self.init_thread = []
        self.query_engine = None
        self.llm = None
        self.embed_model = None
        self.index = None
        self.current_llm = ModelConfig.DEFAULT_LLM
        self.current_embedding = ModelConfig.DEFAULT_EMBEDDING
        self.change_llm = True
        self.change_embedding = True
        self.change_documents = False
        self.question_history = []
        self.running = True

    @staticmethod
    def clear_screen():
        """
        Clear the terminal screen for better readability.

        Uses the appropriate system command based on the operating system
        (cls for Windows, clear for Unix-based systems) to clear the terminal.
        """
        os.system("cls" if os.name == "nt" else "clear")

    def print_welcome(self):
        """
        Display the welcome message and system information.

        Clears the screen and prints a formatted header with the application name,
        current model configurations, and basic usage instructions. This provides
        the user with immediate context about the system's current state.
        """
        self.clear_screen()
        print("=" * 60)
        print(" LLM-based Information Retrieval System ".center(60, "="))
        print("=" * 60)

    def print_models(self):
        """
        Print the current LLM and embedding models.

        :return: None
        :rtype: None
        """
        print(f"Current LLM model: {self.current_llm}")
        print(f"Current embedding model: {self.current_embedding}")
        print("\nType your questions or use a special command (/help for assistance)")
        print("-" * 60)

    @staticmethod
    def show_help():
        """
        Display help information and list all available special commands.

        Prints a formatted list of all special commands defined in SPECIAL_COMMANDS
        along with their descriptions. Also provides basic instructions for asking
        questions to the system.

        This is a static method as it doesn't require access to instance attributes.
        """
        print("\nAvailable commands:")
        for cmd, desc in SPECIAL_COMMANDS.items():
            print(f"  {cmd:<10} - {desc}")
        print("\nTo ask a question, simply type your question and press Enter.")
        print("-" * 60)

    def show_models(self):
        """
        Display available models and allow the user to change model selections.

        Lists all available LLM and embedding models from the ModelConfig settings,
        highlighting the currently selected models. Provides an interactive prompt
        for the user to change either model type, with changes taking effect on
        the next indexing operation.

        Note:
            Changes to model selections are stored in instance variables but
            require re-indexing of documents to take full effect.
        """
        print("\nAvailable LLM models:")
        for key, model in ModelConfig.AVAILABLE_LLM_MODELS.items():
            current = " (current)" if key == self.current_llm else ""
            print(f"  {key:<10} - {model['description']}{current}")

        print("\nAvailable embedding models:")
        for key, model in ModelConfig.AVAILABLE_EMBEDDING_MODELS.items():
            current = " (current)" if key == self.current_embedding else ""
            print(f"  {key:<10} - {model['description']}{current}")

        self._prompt_model_change(
            "LLM",
            ModelConfig.AVAILABLE_LLM_MODELS,
            lambda callback: setattr(self, "current_llm", callback)
            or setattr(self, "change_llm", True),
        )

        self._prompt_model_change(
            "embedding",
            ModelConfig.AVAILABLE_EMBEDDING_MODELS,
            lambda callback: setattr(self, "current_embedding", callback)
            or setattr(self, "change_embedding", True),
        )

        print("\nNote: Changes will take effect during the next indexing.")
        print("-" * 60)

    @staticmethod
    def _prompt_model_change(model_type, available_models, update_callback):
        """
        Prompt the user to change a specific model type.

        Args:
            model_type (str): The type of model ("LLM" or "embedding")
            available_models (dict): Dictionary of available models
            update_callback (callable): Function to call with selected model to update state
        """
        while (
            choice := input(f"\nChange {model_type} model? [name/n]: ").lower()
        ) and choice != "n":
            if choice in available_models:
                update_callback(choice)
                print(f"{model_type} model changed to: {choice}")
                break
            else:
                print(f"Invalid choice: {choice}. Please select a valid model.")

    def show_history(self):
        """
        Display the history of questions asked during the current session.

        Prints a numbered list of all questions that have been asked since
        the application started. If no questions have been asked yet, displays
        an appropriate message. This allows users to review their previous
        interactions with the system.

        Returns:
            None
        """
        if not self.question_history:
            print("\nNo questions in history.")
            return

        print("\nQuestion history:")
        for i, question in enumerate(self.question_history, 1):
            print(f"  {i}. {question}")
        print("-" * 60)

    def handle_command(self, command: str) -> bool:
        """
        Process and execute special commands prefixed with a slash (/).

        Interprets and executes special commands like /help, /quit, /models, etc.
        Each command triggers specific functionality within the application.
        Unknown commands result in an error message and suggestion to use /help.

        Args:
            command (str): The special command to process, including the slash prefix
                          (e.g., "/help", "/quit")

        Returns:
            bool: True if the application should continue running, False if the
                 application should terminate (returned by /quit and /exit commands)
        """
        if command in ["/quit", "/exit"]:
            print("Goodbye!")
            self.running = False
            return False
        elif command == "/help":
            self.show_help()
        elif command == "/models":
            self.show_models()
            if self.change_llm or self.change_embedding:
                self.init()
        elif command == "/clear":
            self.clear_screen()
            self.print_welcome()
            self.print_models()
        elif command == "/history":
            self.show_history()
        elif command == "/config":
            print("\nConfiguration not implemented yet.")
        else:
            print(f"Unknown command: {command}")
            print("Use /help to see available commands.")

        return True

    def process_input(self, user_input: str) -> bool:
        """
        Process user input, handling both commands and natural language questions.

        This is the main input processing method that:
        1. Ignores empty inputs
        2. Routes special commands (starting with /) to the handle_command method
        3. Processes natural language questions by:
           - Adding them to the question history
           - Sending them to the query engine (if configured)
           - Displaying the response or an appropriate error message

        Args:
            user_input (str): The raw input string from the user

        Returns:
            bool: True if the application should continue running, False if it
                 should terminate (when a quit command is processed)

        Raises:
            No exceptions are raised as they are caught internally and reported
            to the user as error messages
        """
        if not user_input.strip():
            return True

        # If it's a special command
        if user_input.startswith("/"):
            return self.handle_command(user_input)

        self.clear_thread()
        if self.init_thread:
            loading_event = threading.Event()
            loading_thread = Thread(
                target=self.show_loading_animation,
                args=(loading_event, "Models are initializing"),
                daemon=True,
            )
            loading_thread.start()

            while self.init_thread:
                self.wait_initialisation()
                self.clear_thread()

            loading_event.set()
            loading_thread.join()

            print("\nModels initialized successfully.")

        # Otherwise, it's a question
        self.question_history.append(user_input)
        if self.query_engine:
            try:
                loading_event = threading.Event()
                loading_thread = Thread(
                    target=self.show_loading_animation,
                    args=(loading_event, "Processing query"),
                    daemon=True,
                )
                loading_thread.start()

                user_input = self.rewrite_query(user_input)
                # todo : put this in a separate history
                print("\nRewritten query:\n", user_input)
                response = self.query_engine.query(user_input)

                loading_event.set()
                loading_thread.join()

                print("\nResponse:\n", response)
            except Exception as e:
                if "loading_event" in locals():
                    loading_event.set()
                    loading_thread.join()
                print(f"\nError processing the query: {str(e)}")
        else:
            print("\nNo query engine is configured. Please load documents first.")

        return True

    @staticmethod
    def show_loading_animation(stop_event, message="Processing"):
        spinner = ["|", "/", "-", "\\"]
        i = 0
        while not stop_event.is_set():
            print(f"\r{message} {spinner[i % len(spinner)]}", end="", flush=True)
            i += 1
            stop_event.wait(0.1)

        print("\r" + " " * (len(message) + 10), end="\r", flush=True)

    def init(self):
        """
        Initialize the models and query engine in a separate thread.

        This method starts a new thread to handle the initialization of
        models and the query engine. It ensures that the main thread remains
        responsive while the initialization is in progress.

        :return: None
        :rtype: None
        """
        self.init_thread.append(Thread(target=self.initialize, daemon=True))
        self.init_thread[-1].start()

    def initialize(self):
        """
        Initializes the models and query engine required for document processing and querying.

        This method sets up the language model (LLM) and embedding model based on the specified
        current configuration. It further initializes a vector index derived from the supplied
        documents using the embedding model. Once the index is created, it is converted into a
        query engine that uses the LLM for processing queries. The initialization status is
        updated to indicate completion.

        :raises RuntimeError: If the initialization process fails at any step.

        """
        if self.change_llm:
            self.change_llm = False
            self.llm = initialize_llm(self.current_llm)

        if self.change_embedding:
            self.change_embedding = False
            self.embed_model = initialize_embedding(self.current_embedding)
            if self.documents:
                self.change_documents = True

        if self.change_documents and self.documents:
            if self.embed_model is not None:
                self.change_documents = False
                self.index = VectorStoreIndex.from_documents(
                    self.documents, embed_model=self.embed_model
                )
            else:
                return self.wait_initialisation() and self.init()

        if self.index is not None:
            self.query_engine = self.index.as_query_engine(llm=self.llm)

        return None

    def run(self):
        """
        Executes the main logic of the application, including initializing resources,
        loading documents, handling user input, and managing threads for specific tasks.
        Coordinates the flow of user interactions and processing as per designated commands
        or queries.

        :return: None
        :rtype: NoneType
        """
        self.print_welcome()

        self.show_models()
        self.print_models()
        self.init()

        # todo : choose files
        paths = [
            "data/text/gps",
        ]

        self.init_thread.append(
            Thread(target=self.load_documents, args=(paths,), daemon=True)
        )
        self.init_thread[-1].start()

        while self.running:
            user_input = input("\nQuestion or command > ")
            self.process_input(user_input)

    def load_documents(self, paths):
        """
        Load documents from multiple paths using threading to improve performance.

        This method creates a separate thread for each path to load documents concurrently.
        It uses ThreadPoolExecutor to manage the threads and collect the results.

        Args:
            paths (List[str]): List of file or directory paths to load documents from

        Returns:
            None: Documents are added to self.documents
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_path = {
                executor.submit(load_documents, path): path for path in paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        self.documents.extend(result)
                        print(f"Loaded documents from {path}")
                    else:
                        print(f"No documents loaded from {path}")
                except Exception as exc:
                    print(f"Error loading documents from {path}: {exc}")

        print(f"\n{len(self.documents)} documents loaded with success.")

        if self.documents:
            self.change_documents = True
            self.init()

    def wait_initialisation(self):
        for index, thread in enumerate(self.init_thread):
            if thread != threading.current_thread():
                thread.join()

    def clear_thread(self):
        for index, thread in enumerate(self.init_thread):
            if not thread.is_alive():
                self.init_thread.pop(index)

    def rewrite_query(self, user_input):
        prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {self.question_history}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
        # todo : delete this
        print(f"\nPrompt:\n{prompt}")
        return chat(
            model=self.current_llm,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
            ],
        ).message.content
