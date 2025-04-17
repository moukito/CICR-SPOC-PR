"""
Utility module for file type detection.

This module provides functionality to identify the type of a given file
based on its content or extension. It serves as a helper for other components
in the system that require file type information to process files appropriately.

Functions:
    detect_file_type: Determines the type of a file and returns its classification.
"""

from .file_type_detector import detect_file_type
