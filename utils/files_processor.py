import os

from llm.document_processor import get_document_processor


def process_directory(directory):
    """
    Process all files in a directory and convert them to document objects.

    Args:
        directory (str): Path to the directory containing files to process

    Returns:
        list: List of document objects created from the files in the directory
    """
    documents = []

    if not os.path.isdir(directory):
        print(f"The directory {directory} do not exist.")
        return documents

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            processor = get_document_processor(file_path)
            document = processor.process_file(file_path)
            documents.append(document)
            print(f"Files successfully loaded: {filename}")
        except Exception as e:
            print(f"An error occurred while loading {filename}: {str(e)}")

    return documents


def process_file(file_path):
    """
    Process an individual file and convert it to a document object.

    Args:
        file_path (str): Path to the file to process

    Returns:
        list: Single-element list containing the document object created from the file
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
    :type path: str
    :return: A list of documents if successfully loaded, or None if no documents
        could be loaded or if the path is invalid.
    :rtype: list or None
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
