#!/usr/bin/env python3
"""Creates a basic question-and-answer loop."""


def main():
    """Continuously accepts questions until the user exits."""
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == "__main__":
    main()
