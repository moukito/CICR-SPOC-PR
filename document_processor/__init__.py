"""
Document Processor Factory Module.

This module provides a factory function for selecting the appropriate document processor
based on the detected file type. It serves as the entry point for the document processing
subsystem, abstracting the selection of specialized processors.

The module integrates the various processor implementations (PDF, DOCX, Text)
and provides a unified interface for the rest of the application to access
document processing capabilities without needing to know the specific processor types.

Functions:
    get_document_processor: Factory function that returns the appropriate
                           document processor for a given file.

Dependencies:
    file_type_detector: Utility module used to detect file types.

This module is a central component of the document processing system,
facilitating the processing of different document types through a common interface.
"""

import os
from .pdf_processor import PDFProcessor
from .text_processor import TextProcessor
from .docx_processor import DocxProcessor
from ..utils.file_type_detector import detect_file_type


def get_document_processor(file_path):
    """
    Returns the appropriate document processor based on the detected file type.

    This factory function examines the provided file path, determines its type using
    the file_type_detector, and instantiates the corresponding processor object.

    Args:
        file_path: Path to the file to be processed

    Returns:
        An appropriate document processor instance

    Raises:
        ValueError: If the file type is not supported by any available processor
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = detect_file_type(file_path)

    if file_type == "pdf":
        return PDFProcessor()
    elif file_type == "text":
        return TextProcessor()
    elif file_type == "docx":
        return DocxProcessor()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
