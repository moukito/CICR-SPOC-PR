""" """

import json

from llama_index.core import VectorStoreIndex

from .utils.files_processor import load_documents
from .utils.initialize_models import initialize_llm, initialize_embedding


def test():
    queries = [
        "De quoi parlent les documents ?",
        "De qui s'agit il ?",
        "Quel est le sujet ?",
    ]
    models = ["mistral", "llama3"]

    documents = []

    path = "data/text/gps"

    documents += load_documents(path)

    print(f"{len(documents)} documents loaded with success.")

    log = {}
    for model in models:
        log[model] = {}

        # Initialise LLM and embedding models
        llm = initialize_llm(model)
        embed_model = initialize_embedding("minilm")

        if not llm or not embed_model:
            print(
                "Initialisation of the model impossible. Please verify the "
                "configuration."
            )
            return

        # Create vector index
        print("Creating vectoriel index...")
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        query_engine = index.as_query_engine(llm=llm)

        responses = []
        for query in queries:
            for i in range(3):
                responses.append(str(query_engine.query(query)))

            log[model][query] = responses

    with open("log.json", "w") as f:
        json.dump(log, f, indent=4)


if __name__ == "__main__":
    test()
