#!/usr/bin/env python3
"""
qeweqwewq asdasd
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
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
