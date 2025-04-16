"""
Processor for plain text documents.

This module provides functionality to extract content from plain text files
and convert it to Document objects for indexing and querying within the
document processing system.

The TextProcessor class handles text file processing with a simple extraction
method that preserves all content. Unlike the PDF and DOCX processors, this
processor is implemented with a static method as text processing requires
no instance state.

Classes:
    TextProcessor: Processes plain text files into Document objects.

Methods:
    TextProcessor.process_file: Static method that reads text files and
                               creates Document objects with appropriate metadata.

This processor is part of the document processing subsystem and is automatically
selected for text files by the get_document_processor factory function.
"""

from llama_index.core import Document


class TextProcessor:
    """
    Processor for handling plain text files extraction.
    Converts text file content to Document objects with appropriate metadata.
    """

    @staticmethod
    def process_file(file_path):
        """
        Extracts content from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            Document: Document containing text and metadata
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return Document(text=content, metadata={"filename": file_path, "type": "text"})
