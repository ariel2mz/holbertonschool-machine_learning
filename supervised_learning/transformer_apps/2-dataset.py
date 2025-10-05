#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    sadsadsadsadsa
    """
    def __init__(self):
        """
        sadasdsadsa
        """
        train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train',
            as_supervised=True)
        validate = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True)
        self.tokenizerpt, self.tokenizeren = self.tokenizedataset(
           train)
        self.datavalid = validate.map(self.tfencode)
        self.datatrain = train.map(self.tfencode)

    def tokenizedataset(self, data):
        """
        sadsads
        """
        ptsen = []
        ensen = []
        for pt, en in data:
            ptsen.append(pt.numpy().decode('utf-8'))
            ensen.append(en.numpy().decode('utf-8'))
        tpt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            clean_up_tokenization_spaces=True, use_fast=True)
        ten = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            clean_up_tokenization_spaces=True, use_fast=True)

        tpt = tpt.train_new_from_iterator(
            ptsen, 2**13)
        ten = ten.train_new_from_iterator(
            ensen, 2**13)

        return tpt, ten

    def encode(self, pt, en):
        """
        sadsadsa
        """
        vsen = self.tokenizeren.vocab_size
        vspt = self.tokenizerpt.vocab_size
        pttok = self.tokenizerpt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False)
        entok = self.tokenizeren.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False)
        pttok = [vspt] + pttok + [vspt + 1]
        entok = [vsen] + entok + [vsen + 1]

        return pttok, entok

    def tfencode(self, pt, en):
        """
        sadsadsa
        """
        resultpt, resulten = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])

        resultpt.set_shape([None])
        resulten.set_shape([None])

        return resultpt, resulten
