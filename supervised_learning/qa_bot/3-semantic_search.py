#!/usr/bin/env python3
"""Performs semantic search on a corpus of documents."""

import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Find the document most semantically similar to a sentence.

    Args:
        corpus_path: Path to the directory containing reference documents.
        sentence: Sentence used to perform the semantic search.

    Returns:
        The text of the most semantically similar document.
    """
    documents = [sentence]

    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(corpus_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())

    embeddings = model(documents)

    similarities = np.inner(embeddings, embeddings)
    closest_index = np.argmax(similarities[0, 1:])

    return documents[closest_index + 1]
