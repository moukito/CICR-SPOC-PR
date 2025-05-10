""" """

import json

from llama_index.core import VectorStoreIndex

from .utils.files_processor import load_documents
from .utils.initialize_models import (
    initialize_llm,
    initialize_embedding,
    initialize_models,
)


def generate_response(log, model, queries, documents, llm, embed_model):
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)

    for query in queries:
        responses = []
        for i in range(5):
            responses.append(str(query_engine.query(query)))

        log[model][query] = responses


def test():
    queries = [
        "Summarize the key topics and themes covered in the document(s).",
        "Provide a concise overview of the main points discussed across the entire document collection.",
        "What is the purpose and scope of the document(s)?",
        "List the main sections or structural parts of the document(s) with a brief description of each.",
        "Give a high-level executive summary suitable for someone who has not read the document(s).",
        "Identify the most important facts, figures, or data points mentioned.",
        "Extract and explain the central argument or thesis of the document(s).",
        "List all definitions or technical terms introduced, along with their explanations.",
        "What are the key insights or findings presented in the document(s), and why are they significant?",
        "Identify all mentioned entities (people, organizations, locations) and describe their roles in the context.",
        "Compare and contrast the views or perspectives presented in different parts of the document(s).",
        "How does the content of the document(s) relate to real-world applications or current issues?",
        "What assumptions are made in the document(s), and are they explicitly justified?",
        "Identify any implicit messages or biases in the document(s).",
        "If you had to teach the content of the document(s) to someone else, how would you structure it?",
        "Generate a list of FAQs (frequently asked questions) based on the content of the document(s).",
        "Create a bullet-point briefing for someone needing to make a decision based on the document(s).",
        "Write a one-paragraph summary suitable for inclusion in a report or dashboard.",
        "What are the possible implications or action points derived from the document(s)?",
        "Suggest three questions a critical reader should ask after reading the document(s).",
    ]

    models = [
        "mistral",
        "llama3",
        "deepseek",
    ]

    embedding_models = [
        "minilm",
        "mpnet",
    ]

    documents = []
    paths = ["data/text/gps"]

    for path in paths:
        documents += load_documents(path)

    print(f"{len(documents)} documents loaded with success.")

    log = {}
    for model in models:
        log[model] = {}

        llm, embed_model = initialize_models(model, "minilm")

        generate_response(log, model, queries, documents, llm, embed_model)

    for embedding_model in embedding_models:
        log[embedding_model] = {}

        llm, embed_model = initialize_models("llama3", embedding_model)

        generate_response(log, embedding_models, queries, documents, llm, embed_model)

    with open("log.json", "w") as f:
        json.dump(log, f, indent=4)


if __name__ == "__main__":
    test()
