#!/usr/bin/env python3
"""
sadsadsadsa
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import os
import numpy as np


def question_answer_model(question, reference):
    """
    sadadasdsa
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    qtokens = tokenizer.tokenize(question)
    rtokens = tokenizer.tokenize(reference)

    if len(qtokens) + len(rtokens) + 3 > 512:
        rtokens = rtokens[:512 - len(qtokens) - 3]

    tokens = ['[CLS]'] + qtokens + ['[SEP]'] + rtokens + ['[SEP]']
    iids = tokenizer.convert_tokens_to_ids(tokens)

    imask = [1] * len(iids)
    sids = [0] * (len(qtokens) + 2) + [1] * (len(rtokens) + 1)

    iids = tf.constant([iids])
    imask = tf.constant([imask])
    sids = tf.constant([sids])

    outputs = model([iids, imask, sids])
    slog, elog = outputs

    i = tf.argmax(slog, axis=1).numpy()[0]
    end = tf.argmax(elog, axis=1).numpy()[0]

    if i == 0 and end == 0:
        return None

    atokens = tokens[i: end + 1]
    answer = tokenizer.convert_tokens_to_string(atokens)

    return answer if answer.strip() else None


def answer_loop(reference):
    """
    asdasdsa
    """
    exit_commands = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input("Q: ").strip()

        if question.lower() in exit_commands:
            print("A: Goodbye")
            break

        answer = question_answer_model(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")


def semantic_search(corpus_path, sentence):
    """
    sadasdsaadsa
    """
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    corpus = []
    file_names = sorted(os.listdir(corpus_path))
    for filename in file_names:
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                corpus.append(f.read())
    corpus_embeddings = embed(corpus)
    sentence_embedding = embed([sentence])
    similarity = np.inner(sentence_embedding, corpus_embeddings)[0]
    best_idx = np.argmax(similarity)

    return corpus[best_idx]


def question_answer(corpus_path):
    """
    dasdasdsa
    """
    exit_commands = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_commands:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = question_answer_model(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
