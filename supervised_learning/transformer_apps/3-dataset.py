#!/usr/bin/env python3
"""
fasfsafas
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    sfaafsa
    """
    def __init__(self, batch_size, max_len):
        """
        asfasfsa
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

        train_dataset = self.raw_data_train.map(self.tf_encode)
        valid_dataset = self.raw_data_valid.map(self.tf_encode)

        def filter_max_len(pt, en):
            """
            safsafs
            """
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        self.data_train = train_dataset
        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        )
        x = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(x)

        self.data_valid = valid_dataset
        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        asfsafsa
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
        afasasfas
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(pt_text)
        en_tokens = self.tokenizer_en.encode(en_text)

        start_token = 2**13
        end_token = 2**13 + 1

        pt_tokens = [start_token] + pt_tokens + [end_token]
        en_tokens = [start_token] + en_tokens + [end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        fasfasfsa
        """
        pt_result, en_result = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_result.set_shape([None])
        en_result.set_shape([None])

        return pt_result, en_result
