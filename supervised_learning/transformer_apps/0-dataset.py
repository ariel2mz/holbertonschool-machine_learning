#!/usr/bin/env python3
"""
qeweqwewq
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    def __init__(self):
        """
            Isadasd
        """
        examples, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tpt, self.ten = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
            adsasd
        """

        abv = "neuralmind/bert-base-portuguese-cased"
        tpt = transformers.AutoTokenizer.from_pretrained(abv)
        ten = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

        pt_sentences = []
        en_sentences = []

        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, 
            vocab_size=2**13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences,
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
