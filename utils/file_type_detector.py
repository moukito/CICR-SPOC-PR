"""
File Type Detection Module

This module provides utilities for detecting and validating file types based on content analysis.
It uses multiple detection mechanisms including MIME type inspection, specialized library validation,
and fallback extension-based analysis to provide accurate file type classification.

The primary use case is for the document processing pipeline to determine which specialized
processor should handle a given file.

Available functions:
    - detect_file_type(file_path): Determines the type of a file using content analysis
"""

import os
import magic
import PyPDF2
import docx
import mimetypes


def detect_file_type(file_path):
    """
    Detects the file type by inspecting its content and MIME type.

    This function uses multiple detection methods to determine file type:
    1. Uses python-magic to identify MIME type
    2. Performs validation checks with specific libraries (PyPDF2, docx)
    3. Falls back to file extension analysis
    4. Attempts text file reading as a last resort

    Args:
        file_path (str): Path to the file to be analyzed

    Returns:
        str: Detected file type, one of:
            - 'pdf': PDF document
            - 'text': Text file (txt, markdown, etc.)
            - 'docx': Microsoft Word document
            - 'unknown': Unrecognized file type

    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)

    if mime_type == "application/pdf":
        try:
            PyPDF2.PdfReader(file_path)
            return "pdf"
        except:
            pass

    elif mime_type == (
        "application/vnd.openxmlformats-officedocument" ".wordprocessingml.document"
    ):
        try:
            docx.Document(file_path)
            return "docx"
        except:
            pass

    elif mime_type.startswith("text/"):
        return "text"

    # If we can't determine the type, check the file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext:
        guess_type = mimetypes.guess_type(file_path)[0]
        if guess_type:
            if "pdf" in guess_type or ext == ".pdf":
                return "pdf"
            elif "text" in guess_type or ext in [".txt", ".md", ".markdown"]:
                return "text"
            elif "word" in guess_type or ext == ".docx":
                return "docx"

    # Last chance try as a text file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(1024)  # Is readable
        return "text"
    except UnicodeDecodeError:
        pass

    return "unknown"
