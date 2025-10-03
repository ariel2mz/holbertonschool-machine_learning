import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    def __init__(self):
        """
            asdasd
        """
        examples, metadata = tfds.load(
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

        tokenizer_pt = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")

        return tokenizer_pt, tokenizer_en
