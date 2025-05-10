""" """

import os

from ..document_processor import get_document_processor


def process_directory(directory):
    """
    Processes all files in the given directory and returns a list of processed documents.

    This function checks whether the given directory exists and processes all files
    within it using a document processor retrieved via the `get_document_processor`
    function. If the directory does not exist, it returns an empty list. For each
    file, it processes the file and adds the processed document to the list.

    :param directory: The path to the directory containing files to be processed.
    :type directory: str
    :return: A list of processed documents, or an empty list if the directory
        does not exist.
    :rtype: list
    """
    documents = []

    if not os.path.isdir(directory):
        print(f"The directory {directory} do not exist.")
        return documents

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            documents += process_file(file_path)
            print(f"Files successfully loaded: {filename}")
        except Exception as e:
            print(f"An error occurred while loading {filename}: {str(e)}")

    return documents


def process_file(file_path):
    """
    Processes a given file and returns a list containing the processed document.

    This function determines the appropriate document processor based on the file
    path, processes the file using the identified processor, and returns the
    processed document wrapped in a list.

    :param file_path: Path to the file to be processed.
    :type file_path: str
    :return: List containing the processed document.
    :rtype: list
    """
    processor = get_document_processor(file_path)
    document = processor.process_file(file_path)
    return [document]


def load_documents(path):
    """
    Loads documents from the given file path. The function handles both files and
    directories. If the provided path is a file, it processes the file. If the
    provided path is a directory, it processes all files within the directory. If
    the path is invalid or no documents could be loaded, it returns None.

    :param path: The file or directory path to load documents from.
    :type path: Str
    :return: A list of documents if successfully loaded, or None if no documents
        could be loaded or if the path is invalid.
    :rtype: List or None
    """
    print(f"Loading documents from {path}...")

    if os.path.isfile(path):
        documents = process_file(path)
    elif os.path.isdir(path):
        documents = process_directory(path)
    else:
        print(f"Invalid path: {path}")
        return None

    if not documents:
        print(
            "No documents could be loaded. Please verify the path of the " "directory."
        )
        return None

    return documents
