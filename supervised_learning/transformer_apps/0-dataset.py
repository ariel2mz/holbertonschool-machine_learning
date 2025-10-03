#!/usr/bin/env python3
import tensorflow_datasets as tfds
import transformers


class Dataset:
    def __init__(self):
        """
            asdasd
        """
        examples, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
            a ver si esto es lo q quiere el cehker
        """
        abv = "neuralmind/bert-base-portuguese-cased"
        tpt = transformers.AutoTokenizer.from_pretrained(abv)
        ten = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

        tpt.model_max_length = 2**13
        ten.model_max_length = 2**13

        return tpt, ten

