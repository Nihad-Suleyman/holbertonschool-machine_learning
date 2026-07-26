#!/usr/bin/env python3
"""Question-answering function using a pre-trained BERT model."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
TOKENIZER_NAME = (
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
model = hub.load(MODEL_URL)


def question_answer(question, reference):
    """
    Find an answer to a question inside a reference document.

    Args:
        question: String containing the question.
        reference: String containing the reference document.

    Returns:
        A string containing the extracted answer, or None if no
        answer is found.
    """
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = (
        ["[CLS]"]
        + question_tokens
        + ["[SEP]"]
        + reference_tokens
        + ["[SEP]"]
    )

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    question_length = len(question_tokens) + 2
    reference_length = len(reference_tokens) + 1

    input_mask = [1] * len(input_ids)
    token_type_ids = (
        [0] * question_length
        + [1] * reference_length
    )

    input_ids = tf.expand_dims(
        tf.convert_to_tensor(input_ids), axis=0
    )
    input_mask = tf.expand_dims(
        tf.convert_to_tensor(input_mask), axis=0
    )
    token_type_ids = tf.expand_dims(
        tf.convert_to_tensor(token_type_ids), axis=0
    )

    outputs = model(
        [input_ids, input_mask, token_type_ids]
    )

    start_logits, end_logits = outputs

    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0]

    if start_index == 0 or end_index < start_index:
        return None

    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer or answer == "[CLS]":
        return None

    return answer
