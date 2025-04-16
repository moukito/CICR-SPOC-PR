"""
Processor for PDF documents.

This module provides functionality to extract text and tables from PDF files
and convert them to Document objects for indexing and querying within the
document processing system.

The PDFProcessor class offers more complex extraction capabilities than the
TextProcessor, handling both textual content and tabular data. It preserves
the page structure and formats tables in a readable way, with page numbers
and table identifiers to maintain context.

Classes:
    PDFProcessor: Processes PDF files into Document objects.

Methods:
    PDFProcessor.process_file: Extracts content from PDF files and creates
                              Document objects with appropriate metadata.
    PDFProcessor.extract_pdf_text_and_tables: Handles the detailed extraction
                                             of text and tables from PDF documents.

Dependencies:
    pdfplumber: Used for PDF content extraction.

This processor is part of the document processing subsystem and is automatically
selected for PDF files by the get_document_processor factory function.
"""

import pdfplumber
from llama_index.core import Document


class PDFProcessor:
    """
    Processor for handling PDF files extraction.
    Converts PDF content to Document objects with appropriate metadata,
    preserving both textual content and tabular data structure.
    """

    def process_file(self, file_path):
        """
        Extracts content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document: Document containing text and metadata
        """
        all_text = self.extract_pdf_text_and_tables(file_path)
        return Document(text=all_text, metadata={"filename": file_path, "type": "pdf"})

    @staticmethod
    def extract_pdf_text_and_tables(pdf_path):
        """
        Extracts text and tables from a PDF file.

        This method processes each page of the PDF document, extracting both
        raw text and tabular data. Each page and table is properly labeled
        with identifying information to maintain document structure.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            str: Formatted string containing all extracted text and tables
        """
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Raw text extraction
                text = page.extract_text() or ""
                all_text += f"\nPage {i + 1}:\n{text}\n"

                # Tables extraction
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    all_text += f"\n[Tableau {t_idx + 1} - Page {i + 1}]\n"
                    for row in table:
                        row_text = " | ".join(
                            str(cell) if cell is not None else "" for cell in row
                        )
                        all_text += row_text + "\n"

        return all_text
