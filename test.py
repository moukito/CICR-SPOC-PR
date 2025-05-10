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
        for i in range(3):
            responses.append(str(query_engine.query(query)))

        log[model][query] = responses


def test():
    queries = [
        "De quoi parlent les documents ?",
        "De qui s'agit il ?",
        "Quel est le sujet ?",
    ]

    models = [
        "mistral",
        "llama3",
    ]

    embedding_models = []

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
