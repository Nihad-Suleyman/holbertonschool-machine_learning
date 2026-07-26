#!/usr/bin/env python3
"""Interactive question-answering loop."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answer questions using information from a reference text.

    Args:
        reference: The reference text used to answer questions.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
