#!/usr/bin/env python3
"""
safasfsafa
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    asfafsagsa
    """
    def __init__(self):
        """
        asffsafas
        """
        examples, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )

        self.raw_data_train = examples['train']
        self.raw_data_valid = examples['validation']

        a, b = self.tokenize_dataset(self.raw_data_train)
        self.tokenizer_pt, self.tokenizer_en = a, b

        self.data_train = self.raw_data_train.map(self.tf_encode)
        self.data_valid = self.raw_data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        afsfasfsa
        """
        abv = "neuralmind/bert-base-portuguese-cased"
        tpt = transformers.AutoTokenizer.from_pretrained(abv)
        ten = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

        pt_sentences = []
        en_sentences = []

        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        tpt = tpt.train_new_from_iterator(
            pt_sentences,
            vocab_size=2**13
        )

        ten = ten.train_new_from_iterator(
            en_sentences,
            vocab_size=2**13
        )

        return tpt, ten

    def encode(self, pt, en):
        """
        fasfsafsa
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(pt_text)
        en_tokens = self.tokenizer_en.encode(en_text)

        start_token = 2**13
        end_token = 2**13 + 1

        pt_tokens = [start_token] + pt_tokens + [end_token]
        en_tokens = [start_token] + en_tokens + [end_token]

        return tf.convert_to_tensor(pt_tokens, dtype=tf.int64),
        tf.convert_to_tensor(en_tokens, dtype=tf.int64)

    def tf_encode(self, pt, en):
        """
        afsfas
        """
        pt_result, en_result = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_result.set_shape([None])
        en_result.set_shape([None])

        return pt_result, en_result
