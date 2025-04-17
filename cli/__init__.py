"""
Package for the command line interface.

This package provides tools and utilities for creating and managing
command-line interfaces within the application. It allows users to
interact with the LLM system through a text-based interface.

Components:
- InteractiveCLI: A class that implements an interactive command-line interface
  for communicating with language models. It handles user input/output and
  manages the conversation flow.
- SPECIAL_COMMANDS: A dictionary of special commands that can be used within
  the CLI environment to perform specific actions like clearing history,
  exiting the application, etc.
"""

from .interactive import InteractiveCLI, SPECIAL_COMMANDS
