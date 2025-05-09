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
from threading import Thread
from typing import Dict, List, Callable, Optional

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
        self.init_thread = None
        self.initialisation_complete = None
        self.query_engine = None
        self.llm = None
        self.embed_model = None
        self.current_llm = ModelConfig.DEFAULT_LLM
        self.current_embedding = ModelConfig.DEFAULT_EMBEDDING
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

        choice = input("\nChange LLM model? [name/n]: ")
        if choice.lower() != "n" and choice in ModelConfig.AVAILABLE_LLM_MODELS:
            self.initialisation_complete = False
            # todo thread this
            self.current_llm = choice
            self.llm = initialize_llm(choice)

            index = VectorStoreIndex.from_documents(
                self.documents, embed_model=self.embed_model
            )

            self.query_engine = index.as_query_engine(llm=self.llm)

            self.initialisation_complete = True
            print(f"LLM model changed to: {choice}")

        choice = input("Change embedding model? [name/n]: ")
        if choice.lower() != "n" and choice in ModelConfig.AVAILABLE_EMBEDDING_MODELS:
            self.initialisation_complete = False
            # todo thread this
            self.current_embedding = choice
            self.embed_model = initialize_embedding(choice)

            index = VectorStoreIndex.from_documents(
                self.documents, embed_model=self.embed_model
            )

            self.query_engine = index.as_query_engine(llm=self.llm)

            self.initialisation_complete = True

            print(f"Embedding model changed to: {choice}")

        print("\nNote: Changes will take effect during the next indexing.")
        print("-" * 60)

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
        elif command == "/clear":
            self.clear_screen()
            self.print_welcome()
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

        if not self.initialisation_complete:
            print("\nModels are still initializing. Please wait...")
            self.init_thread.join()
            print("\nModels initialized successfully.")

        # Otherwise, it's a question
        self.question_history.append(user_input)
        if self.query_engine:
            try:
                response = self.query_engine.query(user_input)
                print("\nResponse:\n", response)
            except Exception as e:
                print(f"\nError processing the query: {str(e)}")
        else:
            print("\nNo query engine is configured. Please load documents first.")

        return True

    def initialize(self):
        """
        Initialize the LLM and embedding models based on configuration settings.

        This method sets up the LLM and embedding models using the specified
        configurations. It is run in a separate thread to allow for asynchronous
        initialization without blocking the main interface.

        Returns:
            None
        """
        self.llm, self.embed_model = initialize_models(
            self.current_llm, self.current_embedding
        )

        index = VectorStoreIndex.from_documents(
            self.documents, embed_model=self.embed_model
        )

        self.query_engine = index.as_query_engine(llm=self.llm)

        self.initialisation_complete = True

    def run(self):
        """ """
        self.print_welcome()

        self.initialisation_complete = False

        paths = ["data/text/gps"]

        # todo thread this
        for path in paths:
            self.documents += load_documents(path)

        print(f"\n{len(self.documents)} documents loaded with success.")

        self.init_thread = Thread(target=self.initialize, daemon=True)
        self.init_thread.start()

        self.print_models()

        while self.running:
            user_input = input("\nQuestion or command > ")
            self.process_input(user_input)
