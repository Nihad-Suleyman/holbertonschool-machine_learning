#!/usr/bin/env python3
"""Answers questions using multiple reference documents."""

qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answer questions using the most relevant document in a corpus.

    Args:
        corpus_path: Path to the directory of reference documents.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
